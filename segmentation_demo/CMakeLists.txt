cmake_minimum_required(VERSION 3.1)

project(segmentation_demo)

set(SOURCES
        src/main.cpp
)

set(LIBRARIES
        cimg
        edison_gpu
)

add_subdirectory(thirdparty)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
