# Set the minimum CMake version
cmake_minimum_required(VERSION 3.16)

# Define the project
project(myk-llama)

# Add the main executable
# minimal example to just load a model then exit
add_executable(myk-llama-simple src/main-simple.cpp)
# more complex example that does some inference from llamacpp examples
add_executable(myk-llama-adv src/main-adv.cpp)
add_executable(super-njam-cli src/main-njam.cpp)

# Add the llama.cpp subdirectory and link the library (if it generates one)
add_subdirectory(../../libs/llama.cpp llama.cpp)

# Optionally link any library built by llamacpp to the main executable
# (Assuming the llamacpp CMakeLists.txt generates a target named `llamacpp-lib`)
# target_link_libraries(myk-llama PRIVATE llamacpp-lib)
target_link_libraries(myk-llama-simple PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(myk-llama-adv PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(super-njam-cli PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
