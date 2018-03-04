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

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
