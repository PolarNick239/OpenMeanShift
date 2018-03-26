#pragma once

#include "common.h"
#include "Platform.h"
#include "Device.h"
#include "Kernel.h"

namespace cl {

    class Engine {
    public:
        const Device_ptr device;

        Engine(Device_ptr device) : device(device), initialized(false) { }
        ~Engine();

        bool init();
        bool ready() const;

        bool compile(const char* source, size_t length, cl_program& program, const char* options=NULL) const;
        Kernel_ptr createKernel(cl_program program, const char* kernel_name) const;
        Kernel_ptr compileKernel(const char* source, size_t length, const char* kernel_name, const char* options=NULL) const;

        cl_mem createBuffer(size_t size, cl_mem_flags flags=CL_MEM_READ_WRITE) const;
        void deallocateBuffer(cl_mem buffer) const;

        void enqueueKernel(Kernel_ptr kernel, unsigned int workDim, const size_t* globalWorkSize, const size_t* localWorkSize, const size_t* globalWorkOffset=NULL, cl_event* event=NULL, const cl_event* waitList=NULL, cl_uint numEventsInWaitList=0) const;

        void writeBuffer(cl_mem buffer, size_t size, const void* ptr) const;
        void readBuffer(cl_mem buffer, size_t size, void* ptr) const;

        void waitForEvents(cl_uint numEvents, const cl_event *eventList) const;

        void finish() const;

    protected:
        cl_context context;
        cl_command_queue queue;
        bool initialized;
    };

    typedef std::shared_ptr<Engine> Engine_ptr;

    Engine_ptr createGPUEngine();
    Engine_ptr createCPUEngine();
    
    class BufferGuard {
    public:
        BufferGuard(cl_mem buffer, Engine_ptr engine) : buffer(buffer), engine(engine)
        {}
        ~BufferGuard()
        {
            engine->deallocateBuffer(buffer);
        }
    protected:
        cl_mem buffer{};
        std::shared_ptr<Engine> engine;
    };

}
