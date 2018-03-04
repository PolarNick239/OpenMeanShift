cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

message("Build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(edison_gpu)
add_subdirectory(segmentation_demo)
