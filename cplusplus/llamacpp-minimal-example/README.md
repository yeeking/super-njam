# Minimal example of using llama cpp from a cpp file

I am working on a C++ project that integrates llamacpp as a runtime for language models. 

I wanted an absolutely minimal example of a CMake project that links against llamacpp and loads a model from a GGUF file.

This is it!

To use it:

```
# clone this project
git clone git@github.com:yeeking/llamacpp-minimal-example.git

# cd into the project folder
cd llamacpp-minimal-example

# clone llama cpp
git clone git@github.com:ggml-org/llama.cpp.git

# generate the project
# dynamic linking - you probably don't want this as the binary will be less portable to others' computers
cmake -B build . 
# static linking: llama cpp is baked into the binary. 
cmake -B build -DBUILD_SHARED_LIBS=OFF . 

# build
cmake --build build --config Debug   -j 10 # number of threads to use

# run with the example supertiny model (which is untrained and just for testing)
./build/myk-llama-simple models/Supertinyllama-107K-F16.gguf
./build/myk-llama-adv -m models/Supertinyllama-107K-F16.gguf

```


