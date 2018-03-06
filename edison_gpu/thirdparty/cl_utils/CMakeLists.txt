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

function(convertIntoHeader sourceFile headerFile arrayName)
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/${headerFile}

            COMMAND xxd -ps ${PROJECT_SOURCE_DIR}/${sourceFile} | fold -w2 | paste -sd' '| sed -e 's/ /, 0x/g' | printf \"0x$$\(cat -\)\" | fold -w120 | printf \"\#include <cstddef>\\n\\nstatic const char ${arrayName}[] = {\\n$$\(cat -\)}\$<SEMICOLON>\\n\\nsize_t ${arrayName}_length = sizeof\(${arrayName}\) / sizeof\(unsigned char\)$<SEMICOLON>\\n\" > ${PROJECT_SOURCE_DIR}/${headerFile}

            DEPENDS ${PROJECT_SOURCE_DIR}/${sourceFile}
    )
endfunction()
