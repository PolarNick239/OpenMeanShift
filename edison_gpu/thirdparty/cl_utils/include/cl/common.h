#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define clew_STATIC
#include "clew.h"

#include <string>
#include <iostream>


#define VERBOSE 0

#if VERBOSE
    #define verbose_cout std::cout
    #define verbose_cerr std::cerr
#else
    #define verbose_cout if (false) std::cout
    #define verbose_cerr if (false) std::cerr
#endif

namespace cl {

    bool initOpenCL();

    std::string getErrorString(cl_int errorCode);

    static bool failure(std::string filename, int line, cl_int returnCode) {
        if (returnCode != CL_SUCCESS) {
            verbose_cerr << "Failure at " << filename << ":" << line << "!\n"
                         << " Error " << returnCode << ": " << getErrorString(returnCode) << std::endl;
            return true;
        } else {
            return false;
        }
    }

    #define OK(call) (!cl::failure(__FILE__, __LINE__, call))
    #define CHECKED(call) if (!OK(call)) throw "Fatal failure!";
    #define CHECKED_FALSE(call) if (!OK(call)) return false;
    #define CHECKED_NULL(call) if (!OK(call)) return nullptr;

    class Version {
    public:

        const int           majorVersion;
        const int           minorVersion;
        const std::string   info;


        Version(const int majorVersion, const int minorVersion, const std::string &info="") :
                majorVersion(majorVersion), minorVersion(minorVersion), info(info) { }

        bool operator<(Version other) const;
    };

    Version parseOCLVersion(std::string version);

}
