#include <climits>
#include <vector>

#include "../../include/caffe/blob.hpp"
#include "../../include/caffe/common.hpp"
#include "../../include/caffe/syncedmem.hpp"
#include "../../include/caffe/util/math_functions.hpp"
#include "../../include/caffe/blob.hpp"
//#include <glog/logging.h>

///部分参考https://www.jianshu.com/p/4289c15b45a0


namespace caffe{

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
        const int width) {
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    Reshape(shape);///调用Reshape(const vector<int> &shape)
}


///注意下面这个Reshpe最为重要， 其余的Reshape方式最后都必须套用这个方法，来改变data_, diff_, count_, shape_...etc
template <typename Dtype> ///template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int> &shape) {
    CHECK_LE(shape.size(), kMAXBlobAxes); ///CHECK_LE 相当于 assert(val1 <= val2)
    count_ = 1; /// protected value
    shape_.resize(shape.size()); ///shape_ 存取数据形状[0]~[3] 共4个元素


    if (!shape_data_ || shape_data_ ->size() < shape.size() * sizeof(int))
    ///如果数据的维度存储位置 不小于传递进来的vector大小
    {
        shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
        ///依照传递进来的vector 大小重新分配 shape_data_内存, reset是指针的方法
    }
    int * shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
    ///shape_data_ 调用方法， 返回可读写指针
    /// static_cast 良性转换， 转换为int*
    for (int i = 0; i < shape.size(); ++i) {
        CHECK_GE(shape[i], 0); ///确保 传入的shape都大于0
        if (count_ != 0 ){
            CHECK_LE(shape[i], INT_MAX / count_ ) << "blob size exceeds INT_MAX";
        } ///确保 shape的元素不超过最大整数极限
            ///#define INT_MAX         2147483647

        ///最后进行赋值
        count_ *= shape[i];
        shape_[i] = shape[i]; ///shape每一个维度都赋值给自己
        shape_data[i] = shape[i];
    }
    if(count_ > capacity_) { ///如果count_超出当前容量， 则扩容
        capacity_ = count_;
        data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        ///data_ 及diff_智能指针重新分配内存
    }
}

template <typename Dtype>///reshape 传入Blobshape类型（定义在caffe.proto)
void Blob<Dtype>::Reshape(const BlobShape & shape){
    CHECK_LE(shape.dim_size(), kMAXBlobAxes); ///确保shape的维度不大于32
    vector<int> shape_vec(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i){///先将shape的每一个维度都赋值
        shape_vec[i] = shape.dim(i);
    }
    Reshape(shape_vec);///执行Reshape(const vector<int> &shape)
}
template <typename Dtype>
///reshape 传入Blob<Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype> & other) {
    Reshape(other.shape());
}




///构造函数
///capacity_ 成员初始化列表方式定义
template <typename Dtype>///自己带入参数初始化
Blob<Dtype>::Blob(const int num, const int channels, const int height, const int width):capacity_(0) {
    Reshape(num, channels, height, width);
}

template <typename Dtype>///依照vector类型初始化
Blob<Dtype>::Blob(const std::vector<int> &shape) : capacity_(0){
    Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const{
    CHECK(shape_data_);
    return (const int*)shape_data_->gpu_data();
    ///shape_data_智能指针指向syncedmem.cpp，表示维度信息存储位置
    /// 调用gpu_data返回const指针
}


template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
    CHECK(data_);
    return (const Dtype*)data_->cpu_data();
    ///data_智能指针指向syncedmem.cpp， 调用cpu_data返回const指针
}


template <typename Dtype> ///set_cpu_data 是syncedmem.hpp中定义的方法
void Blob<Dtype>::set_cpu_data(Dtype *data) {
    CHECK(data);
    ///确保cpu 和 gpu上的数据大小一样
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size ){
    ///如果数据大小 不等于
    ///则分配一个与size大小一样的内存
        data_.reset(new SyncedMemory(size)); ///依照大小重新分配内存
        diff_.reset(new SyncedMemory(size));
        ///智能指针调用reset函数
    }
    data_->set_cpu_data(data);
    ///分配好足够内存之后， data_指针调用函数将数据传入

}


template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
        CHECK(data_);
        return (const Dtype*)data_->gpu_data();
        ///data_智能指针指向syncedmem.cpp， 调用cpu_data返回const指针
    }



template <typename Dtype>///set_gpu_data 也是syncedmem.hpp中定义的方法
void Blob<Dtype>::set_gpu_data(Dtype *data) {
    CHECK(data);
    ///确保cpu 和 gpu上的数据大小一样
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size ){
        ///如果数据大小 不等于
        ///则分配一个与size大小一样的内存
        data_.reset(new SyncedMemory(size)); ///依照大小重新分配内存
        diff_.reset(new SyncedMemory(size));
        ///智能指针调用reset函数
    }
    data_->set_gpu_data(data);
    ///分配好足够内存之后， data_指针调用函数将数据传入

}

////////取得梯度///////////
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const{
    CHECK(diff_);
    return (const Dtype*)diff_ -> cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const{
    CHECK(diff_);
    return (const Dtype*)diff_ -> gpu_data();
}

///取得可读写的数据或梯度

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
    CHECK(diff_);
    return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
    CHECK(diff_);
    return static_cast<Dtype *>(diff_->mutable_gpu_data());

}

///共享Data或Diff
template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob &other) {
        CHECK_EQ(count_, other.count()); ///确保other的大小和目前blob元素总个数一样
        data_ = other.data();///将ohter的赋值给data_，相当于将数据分享
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob &other) {
    CHECK_EQ(count_, other.count());
    diff_ = other.diff();
}


///存在blob中的parameter都会是double or float类型
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }



template <typename Dtype>///梯度更新 data = data - diff
void Blob<Dtype>::Update() {
    ///如何update 取决于目前数据的位置
    switch(data_->head()){ ///data_调用head判断当前状态
        case SyncedMemory::HEAD_AT_CPU:
            ///在CPU上执行运算
            ///caffe_axpy封装在math_functions， axpy : Y = alpha *x + Y
            caffe_axpy<Dtype>(count_, Dtype(-1),
                    static_cast<const Dtype*>(diff_ ->cpu_data()),
                    static_cast<Dtype*>(data_->mutable_cpu_data()));
                    ///static_cast 进行类型转换
            break;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    //在GPU上执行计算
        caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
                static_cast<const Dtype*>(diff_->gpu_data()),
                static_cast<Dtype*>(data_->mutable_gpu_data()));
                ///static_cast 进行类型转换
#else
    NO_GPU
#endif
    break;
    default:
        LOG(FATAL)<< "Syncedmem not initialized.";
    }
}

///unsigned int 和 int 不提供计算
template<>
unsigned int Blob<unsigned int>::asum_data() const{
    NOT_IMPLEMENTED;
}

template<> int Blob<int>::asum_data() const {
    NOT_IMPLEMENTED;
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
    if (!data_) { return 0; } ///如果没有data_，
    switch (data_->head()) {
    case SyncedMemory::HEAD_AT_CPU:
        return caffe_cpu_asum(count_, cpu_data());
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    ///caffe_gpu_asum求x绝对值之和
    return asum;

    }
#else
    NO_GPU;
#endif
    case SyncedMemory::UNINITIALIZED:
        return 0;
    default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
}
    return 0;
}


template <> unsigned int Blob<unsigned int>::asum_diff() const {
    NOT_IMPLEMENTED;
    return 0;
}

template <> int Blob<int>::asum_diff() const {
    NOT_IMPLEMENTED;
    return 0;
}



template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
    if (!diff_) { return 0; }///如果没有data_，
    switch (diff_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            return caffe_cpu_asum(count_, cpu_diff());
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
        {
            Dtype asum;
            caffe_gpu_asum(count_, gpu_diff(), &asum);
            ///caffe_gpu_asum求x绝对值之和
            return asum;
        }
#else
            NO_GPU;
#endif
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
    }
    return 0;
}

template<> unsigned int Blob<unsigned int>::sumsq_data() const {
    NOT_IMPLEMENTED;
    return 0;
}

template <> int Blob<int>::sumsq_data() const {
    NOT_IMPLEMENTED;
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
    Dtype sumsq;
    const Dtype* data;
    if (!data_) {return 0;}
    switch (data_->head()){
    case SyncedMemory::HEAD_AT_CPU: ///CPU上做计算
        data = cpu_data();
        sumsq = caffe_cpu_dot(count_, data, data);
        ///caffe_cpu_dot 计算平方和
        break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
#ifndef CPU_ONLY ///GPU上做计算
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
    case SyncedMemory::UNINITIALIZED:
        return 0;
    default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
    }
    return sumsq;
}
template <>
unsigned int Blob<unsigned  int>::sumsq_diff() const{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
int Blob<int>::sumsq_diff() const {
    NOT_IMPLEMENTED;
    return 0;

}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
    Dtype sumsq;
    const Dtype* diff;
    if (!diff_) {return 0;}
    switch (diff_->head()){
        case SyncedMemory::HEAD_AT_CPU: ///CPU上做计算
            diff = cpu_diff();
            sumsq = caffe_cpu_dot(count_, diff, diff);
            ///caffe_cpu_dot 计算平方和
            break;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY ///GPU上做计算
            diff = gpu_data();
            caffe_gpu_dot(count_, diff, diff, &sumsq);
#else
            NO_GPU;
#endif
            break;
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
    }
    return sumsq;
}

/////////scale_data//////////
template<> void Blob<unsigned int>::scale_data(unsigned int scale_factor){
    NOT_IMPLEMENTED;
}

template<> void Blob<int>::scale_data(int scale_factor) {
    NOT_IMPLEMENTED;
}


template<typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor){
    Dtype* data;
    if(!data_) {return;}
    switch (data_->head()){
    case SyncedMemory::HEAD_AT_CPU:
        data = mutable_cpu_data(); ///mutable 返回可写指针
        caffe_scal(count_, scale_factor, data);
        return;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
    case SyncedMemory::UNINITIALIZED:
        return;
    default:
        LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
    }
}


template<>void Blob<unsigned int>::scale_diff(unsigned int scale_factor){
    NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
    Dtype* diff;
    if (!diff_) { return; }
    switch (diff_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            diff = mutable_cpu_diff();
            caffe_scal(count_, scale_factor, diff);
            return;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
            diff = mutable_gpu_diff();
            caffe_gpu_scal(count_, scale_factor, diff);
            return;
#else
            NO_GPU;
#endif
        case SyncedMemory::UNINITIALIZED:
            return;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
    }
}


template <typename Dtype> ///判断两个维度是否相同
bool Blob<Dtype>::ShapeEquals(const BlobProto &other) {
    if (other.has_num() || other.has_channels() || other.has_height() || other.has_width()){

        return shape_.size() <= 4 &&
               LegacyShape(-4) == other.num() &&
               LegacyShape(-3) == other.channels()&&
               LegacyShape(-2) == other.height() &&
               LegacyShape(-1) == other.width();
    }
    vector<int>other_shape(other.shape().dim_size());
    ///创建一个vector类的other_shape， 大小为传进来的other

    ///将传入的other 每一个元素值赋值给other_shape
    for (int i = 0; i < other.shape().dim_size(); ++i) {
        other_shape[i] = other.shape().dim(i);
    }
    return shape_ == other_shape; ///对照每一个值是否一样， 返回True or False
}

///从某一Blob拷贝数据到当前Blob
template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {

        if (source.count() != count_ || source.shape() != shape_)
        ///如果source blob的结构与当前blob不同， 则执行Reshapelike
        {
            if (reshape) {
                ReshapeLike(source);
            }
            else {
                LOG(FATAL) << "Trying to copy blobs of different sizes.";
            }
        }
        switch (Caffe::mode()) { ///判断Caffe的mode CPU or GPU
            case Caffe::GPU:
                if (copy_diff) {
                    ///调用math_fun里面的caffec_copy函数
                    ///void caffe_copy(const int N, const Dtype* X, Dtype* Y)
                    caffe_copy(count_, source.gpu_diff(),
                            static_cast<Dtype*>(diff_->mutable_gpu_data()));
                }
                else{
                    caffe_copy(count_, source.cpu_data(), static_cast<Dtype*>(data_->mutable_cpu_data()));
                }
                break;
            case Caffe::CPU:
                if (copy_diff) {
                    caffe_copy(count_, source.cpu_diff(),
                            static_cast<Dtype*>(diff_->mutable_cpu_data()));
                }else {
                    caffe_copy(count_, source.cpu_data(),
                    static_cast<Dtype*>(data_-> mutable_cpu_data()));
                }
                break;
            default:
                LOG(FATAL) << "Unknown caffe mode.";
        }
}

template <typename Dtype> ///FromProto 反序列化， 从BlobProto中变成Blob
void Blob<Dtype>::FromProto(const BlobProto & proto, bool reshape){
    if (reshape) { ///如果reshape = True
        vector<int> shape; ///创建vector类shape
        if (proto.has_num() || proto.has_channels() ||
            proto.has_height() || proto.has_width()) {
            shape.resize(4); ///resize成4维度

            ///以下是将传入的proto维度赋值
            shape[0] = proto.num();
            shape[1] = proto.channels();
            shape[2] = proto.height();
            shape[3] = proto.width();
        } else {
            shape.resize(proto.shape().dim_size());///直接resize成和proto一样的维度

            ///利用for loop一个一个维度赋值
            for (int i = 0; i < proto.shape().dim_size(); ++i) {
                shape[i] = proto.shape().dim(i);
            }
        }
        Reshape(shape);
        ///    void Reshape(const std::vector<int> &shape) 或者是
        ///   void Reshape(const BlobShape& shape); ///Blobshape定义在caffe.proto
        }else{
            CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
        }
        // copy data
        Dtype*data_vec = mutable_cpu_data();
        if (proto.double_data_size() > 0) { ///如果数据保存为double 类型data
            CHECK_EQ(count_, proto.double_data_size()); ///确认参数1 and 2是否相同
            for(int i = 0; i < count_; ++i){
                data_vec[i] = proto.double_data(i);
            }
        }else {
            CHECK_EQ(count_, proto.data_size());
            for (int i = 0; i < count_; ++i){
                data_vec[i] = proto.data(i);
            }
        }

        ///下面处理梯度copy
        if (proto.double_diff_size() > 0 ){
            CHECK_EQ(count_, proto.double_diff_size());
            Dtype* diff_vec= mutable_cpu_diff();
            for (int i = 0; i < count_; ++i) {
                diff_vec[i] = proto.double_diff(i);
            }
        }else if (proto.diff_size() > 0){
            CHECK_EQ(count_, proto.diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for (int i = 0; i< count_; ++i) {
                diff_vec[i] = proto.diff(i);
            }
            }

        }






template<> ///序列化 将内存中的Blob保存成Blobproto
void Blob<double>::ToProto(caffe::BlobProto *proto, bool write_diff) const {
    proto -> clear_shape();
    for (int i = 0; i < shape_.size(); ++i){
        proto->mutable_shape() ->add_dim(shape_[i]);
    }
    proto->clear_double_data();
    proto->clear_double_diff();
    const double* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i){
        proto->add_double_data(data_vec[i]);
    }
    if (write_diff) {
        const double * diff_vec = cpu_diff();
        for (int i = 0; i < count_; ++i){
            proto->add_double_diff(diff_vec[i]);
        }
    }

}
template<>
void Blob<float>::ToProto(BlobProto *proto, bool write_diff) const {
    proto -> clear_shape();
    for (int i = 0; i < shape_.size(); ++i){
        proto->mutable_shape()->add_dim(shape_[i]);
    }
    proto->clear_double_data();
    proto->clear_double_diff();
    const float* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i){
        proto->add_double_data(data_vec[i]);
    }
    if (write_diff) {
        const float * diff_vec = cpu_diff();
        for (int i = 0; i < count_; ++i){
            proto->add_double_diff(diff_vec[i]);
        }
    }

}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;
// namespace caffe

}
