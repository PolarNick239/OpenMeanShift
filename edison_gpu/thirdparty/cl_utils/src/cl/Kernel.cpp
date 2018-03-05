#include "cl/Kernel.h"

using namespace cl;

void Kernel::setArg(unsigned int index, size_t size, const void* value) {
    CHECKED(clSetKernelArg(kernel_, index, size, value));
}

cl_kernel Kernel::kernel() const {
    return kernel_;
}
