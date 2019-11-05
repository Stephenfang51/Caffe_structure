#ifndef CAFFE_DETAIL_BLOB_HPP
#define CAFFE_DETAIL_BLOB_HPP

#include <algorithm>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include "syncedmem.hpp"
#include <iostream>
#include "../../include/caffe/proto/caffe.pb.h"
#include <glog/logging.h>

/**
 *
参考
https://www.cnblogs.com/yinheyi/p/5943459.html
https://cloud.tencent.com/developer/article/1394880
 *
 */



const int kMAXBlobAxes = 32; //Blob最大维度为4维， （一般是4维 num, channels, height, width)

//using namespace boost;
//using namespace std;
namespace caffe{

template <typename Dtype>
class Blob {



public:
    Blob()///默认构造函数
            : data_(), diff_(), count_(0), capacity_(0) {}

    ///定义构造函数
    explicit Blob(const int num, const int channels, const int height, const int width);

    explicit Blob(const std::vector<int> &shape);

    void Reshape(const std::vector<int> &shape);

    void Reshape(const BlobShape& shape); ///Blobshape定义在caffe.proto

    void ReshapeLike(const Blob &other);

    inline std::string shape_string() const { ///打印shape的字符串
        std::ostringstream stream ;
        for (int i = 0; i < shape_.size(); ++i) {
            stream << shape_[i] << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
    }

    inline const std::vector<int> &shape() const { return shape_; }
    inline int shape(int index) const {
        return shape_[CanonicalAxisIndex(index)];
    }

    inline int num_axes() const { return shape_.size(); } ///返回blob是多少维度
    inline int count() const { return count_; } ///返回总共多少数据

    inline int count(int start_axis, int end_axis) const {
        ///以下是确保索引值的设定在正常范围
        CHECK_LE(start_axis, end_axis);
        CHECK_GE(start_axis, 0);
        CHECK_GE(end_axis, 0);

        ///区别索引的设定不超过维度
        CHECK_LE(start_axis, num_axes());
        CHECK_LE(end_axis, num_axes());

        int count = 1;
        for (int i = start_axis; i< end_axis; ++i){
            count *= shape(i);
        }
        return count;
    }


    ///计算从某一个指定的元素开始到最后， 带入上面那个函数
    inline int count(int start_axis) const{
        return count(start_axis,  num_axes());
    }




    ///正数索引值不变， 负值改为正的
    inline int CanonicalAxisIndex(int axis_index) const {

        CHECK_GE(axis_index, -num_axes()) ///如果index值大于等于blob的维度
            <<"axis "<< axis_index << " out of range for " << num_axes()
            <<"- D Blob with shape" << shape_string();
        CHECK_LT(axis_index, num_axes()) ///如果index值小于blob的维度
            <<"axis "<< axis_index << " out of range for " << num_axes()
            <<"- D Blob with shape" << shape_string();

        if (axis_index < 0) {
            return axis_index + num_axes(); //将axis_index变成正的
        }
        return axis_index;
    }


    inline int num() const {return LegacyShape(0);}
    inline int channels() const {return LegacyShape(1);}
    inline int height() const {return LegacyShape(2);}
    inline int width() const {return LegacyShape(3);}

    ///求blob 4维中某一维度的值， 并确保不超过4维度
    inline int LegacyShape (int index) const {
        CHECK_LE(num_axes(), 4) ///确保总维度不超过4
            << "Cannot use legacy accesors on Blobs with > 4 axes.";
        CHECK_LT(index, 4);
        CHECK_GE(index, -4);
        if (index >= num_axes() || index <= num_axes()){
            return 1;
        }
        return shape(index);
    }
    /// 下面函数是计算内存偏移量的
    ///比如大小为(2, 3, 5, 5)的Blob，总共占150个位置
    /// (1, 0, 0, 0)是第二个N， 其在内存中的存储位置是75：(1, 0, 0, 0)表示的是第二张图片的第0个通道的第0行0列
    inline int offset(const int n, const int c= 0, const int h = 0, const int w = 0) const
    {

        CHECK_GE(n, 0);
        CHECK_LE(n, num());
        CHECK_GE(channels(), 0);
        CHECK_LE(c, channels());
        CHECK_GE(height(), 0);
        CHECK_LE(h, height());
        CHECK_GE(width(), 0);
        CHECK_LE(w, width());

        return ((n* channels() + c) * height() + h) * width() + w;

    }

    ///传入vector 计算内存偏移量
    inline int offset(const std::vector<int>& indices) const {
        CHECK_LE(indices.size(), num_axes());
        int offset = 0;
        for (int i = 0; i < num_axes(); ++i)
        {
            offset *= shape(i); ///利用for loop的 将shape(0~3)都相乘
            if (indices.size() > i){
                CHECK_GE(indices[i], 0);
                CHECK_LT(indices[i], shape(i));
                offset += indices[i];
            }
        }
        return offset;

    }

    ///从某一Blob拷贝数据到当前Blob
    void CopyFrom(const Blob<Dtype> & source, bool copy_diff = false, bool reshape = false);

    ////访问某个元素
    inline Dtype data_at(const int n, const int c, const int h, const int w) const {
        return cpu_data()[offset(n, c, h, w)];
    }
    ///访问某个元素的梯度
    inline Dtype diff_at(const int n, const int c, const int h, const int w) const{
        return cpu_diff()[offset(n, c, h, w)];
    }

    inline Dtype data_at(const std::vector<int>& index) const {
        return cpu_data()[offset(index)];
    }

    inline Dtype diff_at(const std::vector<int>& index) const {
        return cpu_diff()[offset(index)];
    }
    ///*****返回指向数据的指针******
    ///注意此处用了shared_ptr。并且使用共享内存类，实现了CPU和GPU数据同步，不管数据在CPU还是GPU都可以取出来
    inline const boost::shared_ptr<SyncedMemory>& data() const {
        CHECK(data_);
        return data_;
    }

    inline const boost::shared_ptr<SyncedMemory>& diff() const {
        CHECK(diff_);
        return diff_;
    }

    /// 只读访问cpu数据， *表示返回指针类型的函数
    const Dtype* cpu_data() const;
    /// 设置cpu数据
    void set_cpu_data(Dtype* data);
    ///
    const int* gpu_shape() const;

    const Dtype* gpu_data() const;
    void set_gpu_data(Dtype* data);
    /// 下面读写访问cpu，gpu里面的数据以及梯度
    const Dtype* cpu_diff() const;
    const Dtype* gpu_diff() const;
    Dtype* mutable_cpu_data();
    Dtype* mutable_gpu_data();
    Dtype* mutable_cpu_diff();
    Dtype* mutable_gpu_diff();

    /// 根据梯度更新data_: x = x - learning_rate * tidu(x)
    void Update();
    /// 反序列化，从BlobProto中恢复Blob
    void FromProto(const BlobProto& proto, bool reshape = true);
    /// 序列化，将内存中的Blob对象保存到BlobProto中
    void ToProto(BlobProto* proto, bool write_diff = false) const;

    /// 计算data_，diff_的L1，L2范数
    ///abs
    Dtype asum_data() const;
    Dtype asum_diff() const;

    ///sum suqare
    Dtype sumsq_data() const;
    Dtype sumsq_diff() const;

    /// data_， diff_乘以一个倍数
    void scale_data(Dtype scale_factor);
    void scale_diff(Dtype scale_factor);

    /// 共享另一个Blob的data_, diff_
    void ShareData(const Blob& other);
    void ShareDiff(const Blob& other);

    ///判断维度是否相同
    bool ShapeEquals(const BlobProto& other);

protected:

    ///智能指针指向SyncedMemory 类
    boost::shared_ptr<SyncedMemory> data_; ///数据本身
    boost::shared_ptr<SyncedMemory> diff_; ///数据的derivative
    boost::shared_ptr<SyncedMemory> shape_data_; ///维度信息存储位置 NCHW
    /**
     SyncedMomory 是个类定义在syncedmem.hpp
      */
    std::vector<int> shape_;/// 当前blob的数据形状, e.g. shape_[0]: num, shape_[1]: channels
    int count_;/// 当前blob的数据元素总个数: count_ = shape_[0]*shape_[1]*...*shape_[end]
    int capacity_; ///当前Blob的元素个数（控制动态分配）
    DISABLE_COPY_AND_ASSIGN(Blob);


//    void Reshape(const int num, const int channels, const int height, const int width);
}; //class Blob


} //namespace caffe
#endif //CAFFE_DETAIL_BLOB_HPP
