#include "cl/Platform.h"

#include <sstream>
#include <algorithm>

#define CHECKED_NULL(call) if (!OK(call)) return nullptr;

using namespace cl;
using std::string;
using std::vector;

std::vector<Platform_ptr> cl::getPlatforms() {    
    cl_uint platforms_num;
    CHECKED(clGetPlatformIDs(0, NULL, &platforms_num));

    vector<cl_platform_id> platforms_ids(platforms_num);
    CHECKED(clGetPlatformIDs(platforms_num, platforms_ids.data(), &platforms_num));
    platforms_ids.resize(std::min(platforms_ids.size(), size_t(platforms_num)));

    vector<Platform_ptr> platforms;
    for (auto platform_id : platforms_ids) {
        auto platform = createPlatform(platform_id);
        if (platform) {
            platforms.push_back(platform);
        }
    }
    return platforms;
}

#define getPlatformInfoString(platform_id, param_name, result) {\
    size_t size;\
    CHECKED_NULL(clGetPlatformInfo(platform_id, param_name, 0, NULL, &size));\
    result = string(size, ' ');\
    CHECKED_NULL(clGetPlatformInfo(platform_id, param_name, size, (void *) result.data(), &size));\
    result.resize(size - 1);\
}

Platform_ptr cl::createPlatform(cl_platform_id platform_id) {
    string profile_name, version, name, vendor, extensions;

    getPlatformInfoString(platform_id, CL_PLATFORM_PROFILE, profile_name);
    getPlatformInfoString(platform_id, CL_PLATFORM_VERSION, version);
    getPlatformInfoString(platform_id, CL_PLATFORM_NAME, name);
    getPlatformInfoString(platform_id, CL_PLATFORM_VENDOR, vendor);
    getPlatformInfoString(platform_id, CL_PLATFORM_EXTENSIONS, extensions);

    std::set<string> extensions_set;
    {
        std::stringstream ss(extensions);
        string item;
        while (getline(ss, item, ' ')) {
            if (item.length() > 0) {
                extensions_set.insert(item);
            }
        }
    }

    PlatformProfile profile;
    {
        if (profile_name == "FULL_PROFILE") {
            profile = FULL_PROFILE;
        } else if (profile_name == "EMBEDDED_PROFILE") {
            profile = EMBEDDED_PROFILE;
        } else {
            profile = UNKNOWN_PROFILE;
        }
    }

    return Platform_ptr(new Platform(name, vendor, parseOCLVersion(version), extensions_set, profile, platform_id));
}
