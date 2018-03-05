#pragma once

#include "common.h"

#include <memory>

namespace cl {

    class Kernel {
    public:

        Kernel(cl_program program, cl_kernel kernel) : program(program), kernel_(kernel) {
        }

        void setArg(unsigned int index, size_t size, const void* value);

        cl_kernel kernel() const;

    protected:
        cl_program program;
        cl_kernel kernel_;

    };

    typedef std::shared_ptr<Kernel> Kernel_ptr;

}
