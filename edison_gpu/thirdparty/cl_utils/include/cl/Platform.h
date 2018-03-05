#pragma once

#include <set>
#include <vector>
#include <memory>

#include "common.h"

namespace cl {

    enum PlatformProfile {
        FULL_PROFILE,
        EMBEDDED_PROFILE,
        UNKNOWN_PROFILE,
    };

    class Platform {
    public:

        const std::string               name;
        const std::string               vendor;
        const Version                   version;
        const std::set<std::string>     extensions;
        const PlatformProfile           profile;

        const cl_platform_id            platform_id;

        Platform(const std::string &name, const std::string &vendor, const Version &version,
                 const std::set<std::string> &extensions, const PlatformProfile profile,
                 const cl_platform_id platform_id) : name(name), vendor(vendor), version(version),
                                                       extensions(extensions), profile(profile),
                                                       platform_id(platform_id) { }
    };

    typedef std::shared_ptr<Platform> Platform_ptr;

    std::vector<Platform_ptr> getPlatforms();
    Platform_ptr createPlatform(cl_platform_id);

}
