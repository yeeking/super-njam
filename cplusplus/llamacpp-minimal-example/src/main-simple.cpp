#include <iostream>
#include <string>
#include "llama.h"

/** supress logs https://github.com/ggml-org/llama.cpp/discussions/1758 */
void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) { (void) level; (void) text; (void) user_data; }

int main(int argc, char* argv[]) {
    if (argc > 1) {
        // call this to supress llama's logging
        llama_log_set(llama_log_callback_null, NULL);
        llama_model_params model_params = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(argv[1], model_params);    
        llama_model_free(model);
    } else {
        std::cout << "Usage: myk-llama <path to gguf file>" << std::endl;
    }

    return 0;
}
