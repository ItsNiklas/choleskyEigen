cmake_minimum_required(VERSION 3.16)
project(choleskyEigen)

find_package(Python 3.10 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(choleskyEigenLib cholesky.cpp)

set(CMAKE_CXX_STANDARD 14)

add_library(choleskyEigen cholesky.cpp)
target_link_libraries(choleskyEigen PRIVATE pybind11::embed)

set(EXECUTABLE_OUTPUT_PATH "/home/bauer/choleskyEigen/out")