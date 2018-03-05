cmake_minimum_required(VERSION 3.1)

project(edison_gpu)

set(HEADERS
        src/mean_shift.h
        src/timer.h
        segm/ms.h
        segm/msImageProcessor.h
        segm/RAList.h
        segm/rlist.h
        segm/tdef.h
)

set(SOURCES
        src/ms_filter_multithreaded.cpp
        src/mean_shift.cpp
        segm/ms.cpp
        segm/msImageProcessor.cpp
        segm/RAList.cpp
        segm/rlist.cpp
)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE segm)
target_include_directories(${PROJECT_NAME} PUBLIC src)
