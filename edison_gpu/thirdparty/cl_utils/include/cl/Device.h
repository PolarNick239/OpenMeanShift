#pragma once

#include <set>
#include <vector>
#include <memory>

#include "common.h"
#include "Platform.h"

namespace cl {

    enum DeviceType {
        CPU_DEVICE = CL_DEVICE_TYPE_CPU,
        GPU_DEVICE = CL_DEVICE_TYPE_GPU,
        ACCELERATOR_DEVICE = CL_DEVICE_TYPE_ACCELERATOR,

        ALL_TYPES = CL_DEVICE_TYPE_ALL,
    };

    class Device {
    public:
        const std::string       name;
        const std::string       vendor;
        const unsigned int      vendor_id;
        const Version           device_version;
        const Version           driver_version;

        const size_t            global_mem_size;
        const size_t            max_mem_alloc_size;
        const size_t            local_mem_size;

        const size_t            max_work_group_size;

        const size_t            max_clock_freq;
        const size_t            max_compute_units;

        const size_t            wavefront_size;

        const std::set<std::string> extensions;

        const Platform_ptr      platform;

        const cl_device_id      device_id;

        Device(const std::string &name, const std::string &vendor, const unsigned int vendor_id,
               const Version &device_version, const Version &driver_version, const size_t global_mem_size,
               const size_t max_mem_alloc_size, const size_t local_mem_size, const size_t max_work_group_size,
               const size_t max_clock_freq, const size_t max_compute_units, const size_t wavefront_size, const Platform_ptr &platform,
               const cl_device_id device_id, const std::set<std::string> &extensions) :
                                               name(name), vendor(vendor), vendor_id(vendor_id),
                                               device_version(device_version), driver_version(driver_version),
                                               global_mem_size(global_mem_size),
                                               max_mem_alloc_size(max_mem_alloc_size), local_mem_size(local_mem_size),
                                               max_work_group_size(max_work_group_size),
                                               max_clock_freq(max_clock_freq), max_compute_units(max_compute_units),
                                               wavefront_size(wavefront_size),
                                               extensions(extensions),
                                               platform(platform), device_id(device_id) { }

        void printInfo() const;
    };

    typedef std::shared_ptr<Device> Device_ptr;

    Device_ptr getGPUDevice();
    Device_ptr getCPUDevice();

    std::vector<Device_ptr> getDevices(Platform_ptr platform, DeviceType deviceTypeMask=ALL_TYPES);
    Device_ptr createDevice(Platform_ptr platform, cl_device_id device_id);

}
