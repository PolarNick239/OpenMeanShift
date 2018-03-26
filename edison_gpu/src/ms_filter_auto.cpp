#include "msImageProcessor.h"

#include <cl/Engine.h>
#include <thread>
#include <cmath>
#include <cstring>

void msImageProcessor::NewNonOptimizedFilter_auto(float sigmaS, float sigmaR)
{
    std::vector<cl::Device_ptr> devices;

    if (cl::initOpenCL()) {
        auto platforms = cl::getPlatforms();
        for (auto platform : platforms) {
            auto platform_gpus = cl::getDevices(platform, cl::GPU_DEVICE);
            for (auto device : platform_gpus) {
                devices.push_back(device);
            }
        }
    }

    if (devices.size() > 0 && VERBOSE) {
        verbose_cout << "Using OpenCL GPUs:" << std::endl;
        for (auto device : devices) {
            device->printInfo();
        }
    }

    std::queue<std::pair<size_t, size_t>> queue;
    std::mutex queueMutex;
    std::vector<std::vector<float>> msRawDatas(devices.size(), std::vector<float>(N * L));
    std::vector<std::vector<std::pair<size_t, size_t>>> workProcessed(devices.size() + 1);

    std::vector<std::thread> threads;

    int limit = 64 * 1024;
    for (int i = 0; i < L; i += limit) {
        queue.push(std::pair<size_t, size_t>(i, std::min(L, i + limit)));
    }

    for (int i = 0; i < devices.size(); ++i) {
        threads.emplace_back([this, &sigmaS, &sigmaR, &msRawDatas, &queue, &queueMutex, &workProcessed, &devices, i] {
            NewNonOptimizedFilter_gpu(sigmaS, sigmaR, msRawDatas[i].data(), &queue, &queueMutex, &workProcessed[i], devices[i]);
        });
    }

    // CPU used for calculations only if there are no GPUs or it is only single one,
    // because when GPU is powerful or there are multiple GPUs - CPU only leads to slowdown
    bool useCPU = (devices.size() <= 1);

    if (useCPU) {
        NewNonOptimizedFilter_omp(sigmaS, sigmaR, msRawData, &queue, &queueMutex, &workProcessed[devices.size()]);
    }

    if (devices.size() > 0) {
        verbose_cout << "Devices performance:" << std::endl;
        const int workTotal = L;
        for (int i = 0; i < workProcessed.size(); ++i) {
            bool isCPU = (i == (workProcessed.size() - 1));
            if (!isCPU)
                threads[i].join();
            if (isCPU && !useCPU)
                continue;

            int threadWork = 0;
            for (auto work : workProcessed[i]) {
                threadWork += work.second - work.first;
                if (!isCPU) {
                    memcpy(&msRawData[work.first * N], &msRawDatas[i][work.first * N],
                           (work.second - work.first) * sizeof(float) * N);
                }
            }

            verbose_cout << " - " << (int) (std::round(threadWork * 100.0 / workTotal)) << "% done by "
                         << (isCPU ? "CPU" : devices[i]->name) << std::endl;
        }
    }
}
