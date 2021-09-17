#ifndef FVSAND_CUDA_H
#define FVSAND_CUDA_H

#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cuda.h>

#define FVSAND_GPU_DEVICE __device__
#define FVSAND_GPU_GLOBAL __global__
#define FVSAND_GPU_HOST __host__
#define FVSAND_GPU_HOST_DEVICE __host__ __device__
//#define FVSAND_HAS_GPU CUDA
namespace FVSAND {
namespace gpu {

using gpuDeviceProp_t = cudaDeviceProp;
using gpuError_t = cudaError_t;
constexpr gpuError_t gpuSuccess = cudaSuccess;

inline gpuError_t gpuGetLastError() { return cudaGetLastError(); }
inline const char* gpuGetErrorString(gpuError_t err) { return cudaGetErrorString(err); }

#define FVSAND_CUDA_CHECK_ERROR(call)                                           \
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

#define FVSAND_GPU_CALL(call) cuda ## call
#define FVSAND_GPU_CALL_CHECK(call) FVSAND_CUDA_CHECK_ERROR(FVSAND_GPU_CALL(call))

#define FVSAND_GPU_LAUNCH_FUNC(func, blocks, threads, sharedmem, stream, ...) \
    func<<<blocks, threads, sharedmem, stream>>>(__VA_ARGS__);

template <typename T>
inline T* allocate_on_device(const size_t size)
{
    T* dptr = nullptr;
    FVSAND_CUDA_CHECK_ERROR(cudaMalloc((void**)(&dptr), size));
    return dptr;
}

template <typename T>
inline void copy_to_device(T* dptr, const T* hptr, const size_t size)
{
    FVSAND_CUDA_CHECK_ERROR(cudaMemcpy(dptr, hptr, size, cudaMemcpyHostToDevice));
}

template <typename T>
inline T* push_to_device(const T* hptr, const size_t size)
{
    T* dptr = allocate_on_device<T>(size);
    FVSAND_CUDA_CHECK_ERROR(cudaMemcpy(dptr, hptr, size, cudaMemcpyHostToDevice));
    return dptr;
}

template <typename T>
inline void pull_from_device(T* hptr, T* dptr, const size_t size)
{
    FVSAND_CUDA_CHECK_ERROR(cudaMemcpy(hptr, dptr, size, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void deallocate_device(T** dptr)
{
    FVSAND_CUDA_CHECK_ERROR(cudaFree(static_cast<void*>(*dptr)));
    *dptr = nullptr;
}

template <typename T>
inline void memset_on_device(T* dptr, T val, const size_t sz)
{
  FVSAND_CUDA_CHECK_ERROR(cudaMemset(dptr, val, sz));
}

inline void synchronize()
{
  cudaDeviceSynchronize();
}

} // namespace gpu
} // namespace FVSAND


#endif /* FVSAND_CUDA_H */
