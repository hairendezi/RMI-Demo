cmake_minimum_required(VERSION 3.12)
project(RQ_RLMI_CPP)

set(CMAKE_CXX_STANDARD 14)

# clang and gcc
set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")

# include_directories(alex)
aux_source_directory(src SRC_FILE)

add_executable(main main.cpp ${SRC_FILE})