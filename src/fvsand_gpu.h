#ifndef FVSAND_GPU_H
#define FVSAND_GPU_H

#if defined(FVSAND_HAS_CUDA)
#include "fvsand_cuda.h"
#elif defined(FVSAND_HAS_HIP)
#include "fvsand_hip.h"
#else
#include "fvsand_nogpu.h"
#endif

namespace FVSAND {
namespace gpu {

#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
#define FVSAND_GPU_CHECK_ERROR(call) {                                             \
        FVSAND::gpu::gpuError_t gpu_ierr = (call);                                 \
        if (FVSAND::gpu::gpuSuccess != gpu_ierr) {                                 \
            std::string errStr(std::string("FVSAND GPU error: ") + __FILE__        \
                               + ":" + std::to_string(__LINE__)                   \
                               + ": " + FVSAND::gpu::gpuGetErrorString(gpu_ierr)); \
            throw std::runtime_error(errStr);                                     \
        }}
#else
#define FVSAND_GPU_CHECK_ERROR(call)  (call)
#endif

#define FVSAND_FREE_DEVICE(dptr) { if (dptr) FVSAND::gpu::deallocate_device(&dptr);};

}
}


#endif /* FVSAND_GPU_H */
