cmake_minimum_required(VERSION 3.1)

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/clew/include/clew.h")
    message(SEND_ERROR "clew library submodule was not downloaded! Please, execute:
        git submodule update --init")
endif()

add_subdirectory(clew)
add_subdirectory(cl_utils)
