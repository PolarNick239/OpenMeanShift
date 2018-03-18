cmake_minimum_required(VERSION 2.8.11)

project(cl_utils)

set(HEADERS
        include/cl/common.h
        include/cl/Platform.h
        include/cl/Device.h
        include/cl/Kernel.h
        include/cl/Engine.h
)

set(SOURCES
        src/cl/common.cpp
        src/cl/Platform.cpp
        src/cl/Device.cpp
        src/cl/Kernel.cpp
        src/cl/Engine.cpp
)

include_directories(../clew/include)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} clew)
target_include_directories(${PROJECT_NAME} PUBLIC include)

add_executable(hexdumparray src/utils/hexdumparray.cpp)

function(convertIntoHeader sourceFile headerFile arrayName)
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/${headerFile}

            COMMAND hexdumparray ${PROJECT_SOURCE_DIR}/${sourceFile} ${PROJECT_SOURCE_DIR}/${headerFile} ${arrayName}

            DEPENDS ${PROJECT_SOURCE_DIR}/${sourceFile} hexdumparray
    )
endfunction()
