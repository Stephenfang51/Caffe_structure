#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
#include "mkl.h"
#endif

#include "common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

///分配内存 GPU
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
        ///判断如果是GPU的话
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaMallocHost(ptr, size));///调用cuda的函数 cudaMallocHost
            *use_cuda = true;
            return;
        }
#endif
#ifdef USE_MKL
        *ptr = mkl_malloc(size ? size:1, 64);


#else
        ///CPU 分配内存
        *ptr = malloc(size); ///如果只是CPU， 就用简单的malloc分配
#endif
        *use_cuda = false;
        CHECK(*ptr) << "host allocation of size " << size << " failed";
        ///确保有*ptr
    }


///释放内存
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
        if (use_cuda) {    ///如果是GPU
            CUDA_CHECK(cudaFreeHost(ptr));
            return;
        }
#endif
#ifdef USE_MKL
        mkl_free(ptr);
#else
        ///释放在host上内存
        free(ptr);
#endif
    }


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */

/**
 * 该类别主要用于内存分配及同步between GPu 和 CPU中
    数据分为自身数据和外部数据两种

 */
class SyncedMemory {
    public:
        ///默认构造
        SyncedMemory();
        ///显示构造
        explicit SyncedMemory(size_t size); ///利用给定数据大小来分配内存初始化
        ~SyncedMemory();///析构
        const void* cpu_data(); ///返回CPU const 指针
        void set_cpu_data(void* data); ///外部指定数据时调用
        const void* gpu_data(); ///返回GPU const 指针
        void set_gpu_data(void* data); ///外部指定数据时调用
        void* mutable_cpu_data();///获取CPU数据指针，可以改变数据内容
        void* mutable_gpu_data();///获取GPU数据指针，可以改变数据内容
        enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
        ///四种当前状态， 依照情况会做改变
        SyncedHead head() const { return head_; }///返回当前状态
        size_t size() const { return size_; }

#ifndef CPU_ONLY
        void async_gpu_push(const cudaStream_t& stream);
#endif

    private:
        void check_device();

        void to_cpu();
        void to_gpu();
        void* cpu_ptr_;///CPU侧的数据指针
        void* gpu_ptr_;///GPU侧的数据指针
        size_t size_; ///数据所占用的内存大小
        SyncedHead head_; ///head_有四种状态g可以选择
        bool own_cpu_data_; ///指示cpu_ptr_是否为对象内部调用CaffeMallocHost分配的CPU内存
        bool cpu_malloc_use_cuda_;
        bool own_gpu_data_;/// 指示gpu_ptr_是否为对象内部调用CaffeMallocHost分配的CPU内存
        int device_; ///GPU设备号

        DISABLE_COPY_AND_ASSIGN(SyncedMemory);
        ///函数定义在common.hpp中， 防止指定的类进行复制或是赋值
    };  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
