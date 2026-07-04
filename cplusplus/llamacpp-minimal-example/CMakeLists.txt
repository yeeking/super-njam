cmake_minimum_required(VERSION 3.16)

project(super-njam-cli)

add_executable(super-njam-cli src/main-njam.cpp)

add_subdirectory(../../libs/llama.cpp llama.cpp)

target_link_libraries(super-njam-cli PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
