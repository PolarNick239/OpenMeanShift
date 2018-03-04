cmake_minimum_required(VERSION 3.1)

project(cimg)

# This is thin wrapper around CImg header-only library available at https://github.com/dtschump/CImg

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/CImg/CImg.h")
    message(SEND_ERROR "CImg library submodule was not downloaded! Please, execute:
        git submodule update --init")
endif()

find_package(Threads REQUIRED)
find_package(X11 REQUIRED)

set(HEADERS
        src/images.h
)

set(SOURCES
        src/images.cpp
)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${X11_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PRIVATE CImg)
