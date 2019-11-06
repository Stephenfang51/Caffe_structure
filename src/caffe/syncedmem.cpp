#include "../../include/caffe/common.hpp"
#include "../../include/caffe/syncedmem.hpp"
#include "../../include/caffe/util/math_functions.hpp"

namespace caffe {
    ///默认构造函数
    SyncedMemory::SyncedMemory()
            : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
              own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
    }
///构造函数
///注意head_ = UNINTIALIZED, 尚未真正分配内存
    SyncedMemory::SyncedMemory(size_t size)
            : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
              own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
    }
///析构函数
    SyncedMemory::~SyncedMemory() {
        check_device();///校验当前GPU设备以及gpu_ptr_所指向的设备是不是构造时获取的GPU设备
        if (cpu_ptr_ && own_cpu_data_) {     /// 自己分配的空间自己负责释放
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
            ///调用函数CaffeFreeHost
        }

#ifndef CPU_ONLY
        if (gpu_ptr_ && own_gpu_data_) { ///自己分配的空间自己释放
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
#endif  // CPU_ONLY
    }


    /**
     * -- 理解思路 --
     * 1. 确认目前状态head_
     * 2. uninitialized时候， 为数据分配内存并且初始化， 改变head_状态
     * 3. head_at_gpy时候， 如果cpu指针为Null, 分配内存给cpu指针， 将gpu数据copy到cpu， head_ = SYNCED
     * to_gpu() 概念雷同
     */

    inline void SyncedMemory::to_cpu() {
        check_device();
        switch (head_) {
            case UNINITIALIZED:/// 如果未分配过内存（构造函数后就是UNINITIALIZED
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                ///为数据分配内存, 调用在头文件定义的函数
                caffe_memset(size_, 0, cpu_ptr_);///数据清零
                ///传入cpu_ptr, 以及数据大小size_, 取代的数据为0
                head_ = HEAD_AT_CPU; ///指示为CPU更新了数据
                own_cpu_data_ = true;
                break;
            case HEAD_AT_GPU: ///如果目前数据在GPU
#ifndef CPU_ONLY
                if (cpu_ptr_ == NULL) {
                    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                    own_cpu_data_ = true;
                }
                ///重新分配内存

                ///将gpu_ptr上的内容 copy 到 cpu_ptr_
                caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
                head_ = SYNCED;
                ///指示为已经同步
#else
                NO_GPU;
#endif
                break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
    }

    inline void SyncedMemory::to_gpu() {
        check_device();
#ifndef CPU_ONLY
        switch (head_) {
            case UNINITIALIZED:
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));///分配内存在GPU
                caffe_gpu_memset(size_, 0, gpu_ptr_);///初始化分配好的内存
                head_ = HEAD_AT_GPU; ///改变head_状态
                own_gpu_data_ = true;
                break;
            case HEAD_AT_CPU: ///如果数据已经在CPU了
                if (gpu_ptr_ == NULL) { ///确认GPU数据尚未被分配
                    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));///分配内存在GPU
                    own_gpu_data_ = true;
                }
                caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);///将CPU数据复制到GPU上
                head_ = SYNCED;///head_改为已经同步
                break;
            case HEAD_AT_GPU:
            case SYNCED:
                break;
        }
#else
        NO_GPU;
#endif
    }
    ///返回CPU指针的值
    const void* SyncedMemory::cpu_data() {
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void* data) {
        check_device();
        CHECK(data);
        ///如果本身已经有数据， 则释放
        if (own_cpu_data_) {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;///将CPU指针指向数据
        head_ = HEAD_AT_CPU;///head_状态改变
        own_cpu_data_ = false;///因为自身已经没有数据， 状态改变
    }

    const void* SyncedMemory::gpu_data() {
        check_device();
#ifndef CPU_ONLY
        to_gpu();
        return (const void*)gpu_ptr_;
#else
        NO_GPU;
  return NULL;
#endif
    }

    void SyncedMemory::set_gpu_data(void* data) {
        check_device();
#ifndef CPU_ONLY
        CHECK(data);
        if (own_gpu_data_) { ///如果本身gpu上有数据 先进行释放
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;///将gpu指针重新指向数据
        head_ = HEAD_AT_GPU;///head_改变
        own_gpu_data_ = false;///因为gpu本身已经没有数据， 状态改变
#else
        NO_GPU;
#endif
    }

    void* SyncedMemory::mutable_cpu_data() {
        check_device();
        to_cpu(); ///将数据copy到cpu
        head_ = HEAD_AT_CPU;
        return cpu_ptr_; ///返回的是可写的指针
    }

    void* SyncedMemory::mutable_gpu_data() {
        check_device();
#ifndef CPU_ONLY
        to_gpu();///将数据copy到gpu
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;///返回的是可写的指针
#else
        NO_GPU;
  return NULL;
#endif
    }

#ifndef CPU_ONLY
    ///异步将cpu数据同步到gpu上
    void SyncedMemory::p(const cudaStream_t& stream) {
        check_device();
        CHECK(head_ == HEAD_AT_CPU); ///確認数据当前的位置在cpu
        if (gpu_ptr_ == NULL) { ///确认gpu指针没有东西
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            own_gpu_data_ = true;
        }
        ///cudaMemcpyKind 指定复制方向 ex Host to Device
        const cudaMemcpyKind put = cudaMemcpyHostToDevice; ///将数据从host copy to gpu
        CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
        ///cudaMemcpyAsync 异步传输
        // Assume caller will synchronize on the stream before use
        head_ = SYNCED; ///状态为同步
    }
#endif

    ///校验当前GPU设备以及gpu_ptr_所指向的设备是不是构造时获取的GPU设备
    void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
        int device;
        cudaGetDevice(&device);
        CHECK(device == device_);
        if (gpu_ptr_ && own_gpu_data_) {
            cudaPointerAttributes attributes; ///cudaPointerAttributes是结构struct
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
            CHECK(attributes.device == device_);
            ///确认attributes结构中的int device是否为构造时的GPU设备
  }
#endif
#endif
    }

}  // namespace caffe

