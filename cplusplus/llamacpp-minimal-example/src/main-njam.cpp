#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void print_usage(const char * argv0) {
    std::fprintf(
        stderr,
        "Usage: %s -m model.gguf -p prompt.njam [-o output.txt] [-n n_predict] [-ngl n_gpu_layers]\n",
        argv0);
}

static std::string read_file(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open input file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt_path;
    std::string output_path;
    int n_predict = 96;
    int ngl = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt_path = argv[++i];
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || prompt_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string prompt = read_file(prompt_path);

    llama_model_params model_params = llama_model_default_params();
    ggml_backend_dev_t cpu_devices[2] = { nullptr, nullptr };
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_devices[0] = dev;
            break;
        }
    }
    if (cpu_devices[0] != nullptr) {
        model_params.devices = cpu_devices;
    }
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        std::fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "Prompt tokenization failed.\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "Prompt tokenization failed.\n");
        llama_model_free(model);
        return 1;
    }

    const uint32_t trained_ctx = llama_model_n_ctx_train(model);
    if (trained_ctx > 0 && static_cast<uint32_t>(n_prompt + n_predict + 1) > trained_ctx) {
        const int max_predict = std::max(1, static_cast<int>(trained_ctx) - n_prompt - 1);
        if (max_predict <= 0) {
            std::fprintf(stderr, "Prompt is too long for model context (%u tokens).\n", trained_ctx);
            llama_model_free(model);
            return 1;
        }
        n_predict = max_predict;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict + 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    ctx_params.offload_kqv = false;
    ctx_params.op_offload = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        std::fprintf(stderr, "Failed to create llama context.\n");
        llama_model_free(model);
        return 1;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string generated = prompt;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    int n_decode = 0;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed.\n");
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        n_pos += batch.n_tokens;
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buffer[256];
        const int piece_size = llama_token_to_piece(vocab, new_token, buffer, sizeof(buffer), 0, true);
        if (piece_size < 0) {
            std::fprintf(stderr, "Failed to decode sampled token.\n");
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        generated.append(buffer, piece_size);
        batch = llama_batch_get_one(&new_token, 1);
        ++n_decode;
    }

    if (!output_path.empty()) {
        std::ofstream output(output_path);
        if (!output) {
            std::fprintf(stderr, "Failed to open output path: %s\n", output_path.c_str());
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        output << generated;
    } else {
        std::printf("%s\n", generated.c_str());
    }

    std::fprintf(stderr, "Decoded %d tokens.\n", n_decode);
    llama_perf_sampler_print(sampler);
    llama_perf_context_print(ctx);

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
