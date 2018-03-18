#include "msImageProcessor.h"

#include <cl/Engine.h>
#include <thread>
#include <cmath>

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
        const int index = i;
        threads.emplace_back([this, &sigmaS, &sigmaR, &msRawDatas, &queue, &queueMutex, &workProcessed, &index, &devices] {
            NewNonOptimizedFilter_gpu(sigmaS, sigmaR, msRawDatas[index].data(), &queue, &queueMutex, &workProcessed[index], devices[index]);
        });
    }

    NewNonOptimizedFilter_omp(sigmaS, sigmaR, msRawData, &queue, &queueMutex, &workProcessed[devices.size()]);

    if (devices.size() > 0) {
        verbose_cout << "Devices performance:" << std::endl;
        const int workTotal = L;
        for (int i = 0; i < workProcessed.size(); ++i) {
            bool isCPU = (i == (workProcessed.size() - 1));
            if (!isCPU)
                threads[i].join();

            int threadWork = 0;
            for (auto work : workProcessed[i]) {
                threadWork += work.second - work.first;
                if (!isCPU) {
#pragma omp parallel for schedule(static)
                    for (int k = work.first; k < work.second; ++k) {
                        for (int j = 0; j < N; ++j) {
                            msRawData[k * N + j] = msRawDatas[i][k * N + j];
                        }
                    }
                }
            }

            verbose_cout << " - " << (int) (std::round(threadWork * 100.0 / workTotal)) << "% done by "
                         << (isCPU ? "CPU" : devices[i]->name) << std::endl;
        }
    }
}
