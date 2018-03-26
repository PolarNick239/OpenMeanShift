#include "msImageProcessor.h"

#include <cl/Engine.h>
#include "timer.h"

#include "ms_filter_opencl_kernel_cl.h"

void msImageProcessor::NewNonOptimizedFilter_gpu(float sigmaS, float sigmaR,
                                                 float* msRawDataRes, std::queue<std::pair<size_t, size_t>>* workQueue, std::mutex* queueLock, std::vector<std::pair<size_t, size_t>>* workProcessed,
                                                 cl::Device_ptr device)
{
    int WORKGROUP_SIZE = 128;

    std::queue<std::pair<size_t, size_t>> tmpQueue;
    std::vector<std::pair<size_t, size_t>> tmpWorkProcessed;
    std::mutex tmpMutex;

    if (msRawDataRes == nullptr && workQueue == nullptr && queueLock == nullptr && workProcessed == nullptr && !device) {
        msRawDataRes = msRawData;
        workQueue = &tmpQueue;
        queueLock = &tmpMutex;
        workProcessed = &tmpWorkProcessed;
        tmpQueue.push(std::pair<int, int>(0, L));

        if (!cl::initOpenCL()) {
            throw std::runtime_error("OpenCL initialization failed!");
        }

        device = cl::getGPUDevice();
        if (!device) {
            device = cl::getCPUDevice();
            if (!device) {
                throw std::runtime_error("No OpenCL devices!");
            }
            verbose_cout << "Using platform: " << device->platform->name << std::endl;
            WORKGROUP_SIZE = std::max(32, (int) device->max_compute_units * 4);
        }
        verbose_cout << "Using device: " << device->name << " with " << device->max_compute_units
                     << " max compute units" << std::endl;
    }

    //make sure that a lattice height and width have
    //been defined...
    if (!height) {
        ErrorHandler("msImageProcessor", "LFilter", "Lattice height and width are undefined.");
        return;
    }

    //re-assign bandwidths to sigmaS and sigmaR
    if (((h[0] = sigmaS) <= 0) || ((h[1] = sigmaR) <= 0)) {
        ErrorHandler("msImageProcessor", "Segment", "sigmaS and/or sigmaR is zero or negative.");
        return;
    }

    //define input data dimension with lattice
    int lN = N + 2;

    // Traverse each data point applying mean shift
    // to each data point

    // let's use some temporary data
    float *sdata;
    std::vector<float> sdata_data(lN * L);
    sdata = sdata_data.data();

    // copy the scaled data
    int idxs, idxd;
    idxs = idxd = 0;
    if (N == 3) {
        for (int i = 0; i < L; i++) {
            sdata[idxs++] = (i % width) / sigmaS;
            sdata[idxs++] = (i / width) / sigmaS;
            sdata[idxs++] = data[idxd++] / sigmaR;
            sdata[idxs++] = data[idxd++] / sigmaR;
            sdata[idxs++] = data[idxd++] / sigmaR;
        }
    } else if (N == 1) {
        for (int i = 0; i < L; i++) {
            sdata[idxs++] = (i % width) / sigmaS;
            sdata[idxs++] = (i / width) / sigmaS;
            sdata[idxs++] = data[idxd++] / sigmaR;
        }
    } else {
        for (int i = 0; i < L; i++) {
            sdata[idxs++] = (i % width) / sigmaS;
            sdata[idxs++] = (i % width) / sigmaS;
            for (int j = 0; j < N; j++)
                sdata[idxs++] = data[idxd++] / sigmaR;
        }
    }
    // index the data in the 3d buckets (x, y, L)
    int *buckets;
    int *slist;
    std::vector<int> slist_data(L);
    slist = slist_data.data();

    float sMins; // just for L
    float sMaxs[3]; // for all
    sMaxs[0] = width / sigmaS;
    sMaxs[1] = height / sigmaS;
    sMins = sMaxs[2] = sdata[2];
    idxs = 2;
    float cval;
    for (int i = 0; i < L; i++) {
        cval = sdata[idxs];
        if (cval < sMins)
            sMins = cval;
        else if (cval > sMaxs[2])
            sMaxs[2] = cval;

        idxs += lN;
    }

    int cBuck1, cBuck2, cBuck3, cBuck;
    const int nBuck1 = (int) (sMaxs[0] + 3);
    const int nBuck2 = (int) (sMaxs[1] + 3);
    const int nBuck3 = (int) (sMaxs[2] - sMins + 3);
    std::vector<int> buckets_data(nBuck1 * nBuck2 * nBuck3);
    buckets = buckets_data.data();
    for (int i = 0; i < (nBuck1 * nBuck2 * nBuck3); i++)
        buckets[i] = -1;

    idxs = 0;
    for (int i = 0; i < L; i++) {
        // find bucket for current data and add it to the list
        cBuck1 = (int) sdata[idxs] + 1;
        cBuck2 = (int) sdata[idxs + 1] + 1;
        cBuck3 = (int) (sdata[idxs + 2] - sMins) + 1;
        cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

        slist[i] = buckets[cBuck];
        buckets[cBuck] = i;

        idxs += lN;
    }
    // done indexing/hashing

    cl::Engine_ptr engine(new cl::Engine(device));
    engine->init();

    cl::Kernel_ptr kernel;

    {
        std::string defines = std::string("")
                              + " -D WORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE)
                              + " -D WAVEFRONT_SIZE=" + std::to_string(engine->device->wavefront_size)
                              + " -D N=" + std::to_string(N)
                              + " -D EPSILON=" + std::to_string(EPSILON) + "f"
                              + " -D LIMIT=" + std::to_string(LIMIT)
                              + " -D sigmaS=" + std::to_string(sigmaS) + "f"
                              + " -D sigmaR=" + std::to_string(sigmaR) + "f"
        ;
        performance_timer timer;
        kernel = engine->compileKernel(mean_shift_kernel, mean_shift_kernel_length, "meanShiftFilter", defines.data());
        if (!kernel)
            throw std::runtime_error("OpenCL kernel compilation failed!");
        verbose_cout << "Kernel compiled in " << timer.elapsed() << " s!" << std::endl;
    }

    cl_mem buf_sdata        = engine->createBuffer(lN * L * sizeof(cl_float),                 CL_MEM_READ_ONLY);  cl::BufferGuard buf_sdata_guard    (buf_sdata,     engine);
    cl_mem buf_buckets      = engine->createBuffer(nBuck1 * nBuck2 * nBuck3 * sizeof(cl_int), CL_MEM_READ_ONLY);  cl::BufferGuard buf_buckets_guard  (buf_buckets,   engine);
    cl_mem buf_weightMap    = engine->createBuffer(L * sizeof(cl_float),                      CL_MEM_READ_ONLY);  cl::BufferGuard buf_weightMap_guard(buf_weightMap, engine);
    cl_mem buf_slist        = engine->createBuffer(L * sizeof(cl_int),                        CL_MEM_READ_ONLY);  cl::BufferGuard buf_slist_guard    (buf_slist,     engine);
    cl_mem buf_msRawData    = engine->createBuffer(N * L * sizeof(cl_float),                  CL_MEM_WRITE_ONLY); cl::BufferGuard buf_msRawData_guard(buf_msRawData, engine);

    engine->writeBuffer(buf_sdata,     lN * L * sizeof(cl_float),                 sdata);
    engine->writeBuffer(buf_buckets,   nBuck1 * nBuck2 * nBuck3 * sizeof(cl_int), buckets);
    engine->writeBuffer(buf_weightMap, L * sizeof(cl_float),                      weightMap);
    engine->writeBuffer(buf_slist,     L * sizeof(cl_int),                        slist);

    {
        unsigned int i = 0;
        kernel->setArg(i++, sizeof(cl_mem), &buf_sdata);
        kernel->setArg(i++, sizeof(cl_mem), &buf_buckets);
        kernel->setArg(i++, sizeof(cl_mem), &buf_weightMap);
        kernel->setArg(i++, sizeof(cl_mem), &buf_slist);
        kernel->setArg(i++, sizeof(cl_mem), &buf_msRawData);
        kernel->setArg(i++, sizeof(int),    &L);
        kernel->setArg(i++, sizeof(int),    &width);
        kernel->setArg(i++, sizeof(int),    &height);
        kernel->setArg(i++, sizeof(float),  &sMins);
        kernel->setArg(i++, sizeof(int),    &nBuck1);
        kernel->setArg(i++, sizeof(int),    &nBuck2);
        kernel->setArg(i++, sizeof(int),    &nBuck3);

        size_t localWorkSize[3];
        size_t globalWorkOffset[3];
        size_t globalWorkSize[3];

        localWorkSize[0] = 1;
        localWorkSize[1] = WORKGROUP_SIZE;

        verbose_cout << "Kernel launched..." << std::endl;

        performance_timer timer;

        int limit = 64 * 1024;

        cl_event event_prev_launch = NULL;

        while (true)
        {
            int workFrom;
            int workTo;
            {
                std::lock_guard<std::mutex> guard(*queueLock);
                if (workQueue->size() == 0) {
                    break;
                }
                auto work = workQueue->front();
                workFrom = work.first;
                workTo = work.second;
                workProcessed->push_back(work);
                workQueue->pop();
            }
            for (int offset = workFrom; offset < workTo; offset += limit) {
                globalWorkOffset[0] = offset;
                globalWorkOffset[1] = 0;
                globalWorkSize[0] = std::min(L - offset, limit);
                globalWorkSize[1] = WORKGROUP_SIZE;

                cl_event event_cur_launch = NULL;
                engine->enqueueKernel(kernel, 2, globalWorkSize, localWorkSize, globalWorkOffset, &event_cur_launch);
                if (event_prev_launch != NULL) {
                    engine->waitForEvents(1, &event_prev_launch);
                }

                event_prev_launch = event_cur_launch;
            }
        }
        engine->finish();

        verbose_cout << "Kernel executed in " << timer.elapsed() << " s" << std::endl;
    }

    performance_timer timer;
    verbose_cout << "Fetching result data..." << std::endl;
    engine->readBuffer(buf_msRawData, N * L * sizeof(cl_float), msRawDataRes);
    verbose_cout << "Results retrieved in " << timer.elapsed() << " s!" << std::endl;
}
