#include "cl/Device.h"

#include <sstream>
#include <algorithm>

using namespace cl;
using std::vector;
using std::string;

std::vector<Device_ptr> cl::getDevices(Platform_ptr platform, DeviceType deviceTypeMask) {
    vector<Device_ptr> devices;

    cl_uint devices_num;
    cl_int code = clGetDeviceIDs(platform->platform_id, deviceTypeMask, 0, NULL, &devices_num);
    if (code == CL_DEVICE_NOT_FOUND) {
        return devices;
    }
    CHECKED(code);

    vector<cl_device_id> devices_ids(devices_num);
    CHECKED(clGetDeviceIDs(platform->platform_id, deviceTypeMask, devices_num, devices_ids.data(), &devices_num));
    devices_ids.resize(std::min(devices_ids.size(), size_t(devices_num)));

    for (auto device_id : devices_ids) {
        auto device = createDevice(platform, device_id);
        if (device) {
            devices.push_back(device);
        }
    }
    return devices;
}

#define getDeviceInfoString(device_id, param_name, result) {\
    size_t size;\
    CHECKED_NULL(clGetDeviceInfo(device_id, param_name, 0, NULL, &size));\
    result = string(size, ' ');\
    CHECKED_NULL(clGetDeviceInfo(device_id, param_name, size, (void *) result.data(), &size));\
    result.resize(size - 1);\
}

Device_ptr cl::createDevice(Platform_ptr platform, cl_device_id device_id) {
    string name, vendor, device_version, driver_version, extensions;
    unsigned int vendor_id;
    cl_ulong global_mem_size, max_mem_alloc_size, local_mem_size;
    size_t max_work_group_size;
    cl_uint max_clock_freq, max_compute_units, wavefront_size;

    getDeviceInfoString(device_id, CL_DEVICE_NAME, name);
    getDeviceInfoString(device_id, CL_DEVICE_VENDOR, vendor);
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL));

    getDeviceInfoString(device_id, CL_DEVICE_VERSION, device_version);
    getDeviceInfoString(device_id, CL_DRIVER_VERSION, driver_version);

    getDeviceInfoString(device_id, CL_DEVICE_EXTENSIONS, extensions);
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL));
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL));
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL));
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL));
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_freq, NULL));
    CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL));

    int driver_major_version = -1;
    int driver_minor_version = -1;
    {
        //<major_version.minor_version>
        std::stringstream ss(driver_version);
        std::string item;
        int item_i = 0;
        while (getline(ss, item, '.')) {
            if (item_i == 0) {
                driver_major_version = atoi(item.data());
            } else if (item_i == 1) {
                driver_minor_version = atoi(item.data());
                break;
            }
            item_i++;
        }
    }

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

    if (extensions_set.count("cl_amd_device_attribute_query")) {
        #define CL_DEVICE_WAVEFRONT_WIDTH_AMD 0x4043
        CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(cl_uint), &wavefront_size, NULL));
    } else if (extensions_set.count("cl_nv_device_attribute_query")) {
        #define CL_DEVICE_WARP_SIZE_NV 0x4003
        CHECKED_NULL(clGetDeviceInfo(device_id, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &wavefront_size, NULL));
    } else {
        wavefront_size = 1;
    }

    return Device_ptr(new Device(name, vendor, vendor_id,
                                 parseOCLVersion(device_version), Version(driver_major_version, driver_minor_version),
                                 global_mem_size, max_mem_alloc_size, local_mem_size, max_work_group_size,
                                 max_clock_freq, max_compute_units, wavefront_size,
                                 platform, device_id,
                                 extensions_set));
}

Device_ptr cl::getGPUDevice() {
    Device_ptr best_device;
    auto platforms = getPlatforms();
    for (auto platform : platforms) {
        auto devices = getDevices(platform, GPU_DEVICE);
        for (auto device : devices) {
            if (!best_device || device->max_compute_units > best_device->max_compute_units) {
                best_device = device;
            }
        }
    }
    return best_device;
}

Device_ptr cl::getCPUDevice() {
    Device_ptr best_device;
    auto platforms = getPlatforms();
    for (auto platform : platforms) {
        auto devices = getDevices(platform, CPU_DEVICE);
        for (auto device : devices) {
            if (!best_device || device->max_compute_units > best_device->max_compute_units) {
                best_device = device;
            }
        }
    }
    return best_device;
}

void Device::printInfo() const {
    std::cout << "Platform: " << platform->name << " " << platform->vendor << std::endl;
    std::cout << "Device:   " << name << " " << vendor << " OpenCL " << device_version.majorVersion << "." << device_version.minorVersion;
    if (!device_version.info.empty()) {
        std::cout << " (" << device_version.info << ")" << std::endl;
    } else {
        std::cout << std::endl;
    }
    std::cout << "          Driver:   " << driver_version.majorVersion << "." << driver_version.minorVersion << std::endl;
    std::cout << "          Max compute units " << max_compute_units << ", memory size " << (global_mem_size >> 20) << " MB, max frequency " << max_clock_freq << " MHz" << std::endl;
}
