#ifndef FVSAND_HIP_H
#define FVSAND_HIP_H

#include <cstdlib>
#include <string>
#include <stdexcept>
#include <hip/hip_runtime.h>

#define FVSAND_GPU_DEVICE __device__
#define FVSAND_GPU_GLOBAL __global__
#define FVSAND_GPU_HOST __host__
#define FVSAND_GPU_HOST_DEVICE __host__ __device__
#define FVSAND_HAS_GPU HIP

namespace FVSAND {
namespace gpu {

using gpuDeviceProp_t = hipDeviceProp_t;
using gpuError_t = hipError_t;
constexpr gpuError_t gpuSuccess = hipSuccess;

inline gpuError_t gpuGetLastError() { return hipGetLastError(); }
inline const char* gpuGetErrorString(gpuError_t err) { return hipGetErrorString(err); }

#define FVSAND_HIP_CHECK_ERROR(call)                                            \
  do {                                                                         \
    FVSAND::gpu::gpuError_t gpu_ierr = (call);                                  \
    if (FVSAND::gpu::gpuSuccess != gpu_ierr) {                                  \
      std::string err_str(                                                     \
        std::string("FVSAND GPU error: ") + __FILE__ + ":" +                    \
        std::to_string(__LINE__) + ": " +                                      \
        FVSAND::gpu::gpuGetErrorString(gpu_ierr));                              \
      throw std::runtime_error(err_str);                                       \
    }                                                                          \
  } while (0)

#define FVSAND_GPU_CALL(call) hip ## call
#define FVSAND_GPU_CALL_CHECK(call) FVSAND_HIP_CHECK_ERROR(FVSAND_GPU_CALL(call))

#define FVSAND_GPU_LAUNCH_FUNC(func, blocks, threads, sharedmem, stream, ...) \
  hipLaunchKernelGGL(func, blocks, threads, sharedmem, stream, __VA_ARGS__);

template <typename T>
inline T* allocate_on_device(const size_t size)
{
    T* dptr = nullptr;
    FVSAND_HIP_CHECK_ERROR(hipMalloc((void**)(&dptr), size));
    return dptr;
}

template <typename T>
inline T* allocate_host_pinned(const size_t size)
{
    T* dptr = nullptr;
    FVSAND_CUDA_CHECK_ERROR(hipHostMalloc((void**)(&dptr), size));
    return dptr;
}


template <typename T>
inline void copy_to_device(T* dptr, const T* hptr, const size_t size)
{
    FVSAND_HIP_CHECK_ERROR(hipMemcpy(dptr, hptr, size, hipMemcpyHostToDevice));
}

template <typename T>
inline T* push_to_device(const T* hptr, const size_t size)
{
    T* dptr = allocate_on_device<T>(size);
    FVSAND_HIP_CHECK_ERROR(hipMemcpy(dptr, hptr, size, hipMemcpyHostToDevice));
    return dptr;
}

template <typename T>
inline void pull_from_device(T* hptr, T* dptr, const size_t size)
{
    FVSAND_HIP_CHECK_ERROR(hipMemcpy(hptr, dptr, size, hipMemcpyDeviceToHost));
}

template <typename T>
inline void deallocate_device(T** dptr)
{
    FVSAND_HIP_CHECK_ERROR(hipFree(static_cast<void*>(*dptr)));
    *dptr = nullptr;
}

template < typename T>
inline void deallocate_host_pinned(T** ptr)
{
    FVSAND_CUDA_CHECK_ERROR(hipHostFree(static_cast<void*>(*ptr)));
    *ptr = nullptr;
}


template <typename T>
inline void memset_on_device(T* dptr, T val, const size_t sz)
{
  FVSAND_HIP_CHECK_ERROR(hipMemset(dptr, val, sz));
}

inline void synchronize()
{
  hipDeviceSynchronize();
}

} // namespace gpu
} // namespace FVSAND


#endif /* FVSAND_HIP_H */
