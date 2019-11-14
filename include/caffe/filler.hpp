//
// Created by StephenFang on 2019/11/14.
//Fillers 是随机数产生器， 可以用特定的算法填满一个blob类型的数据，只用来在
//初始化的时候使用，跟gpu无关

#ifndef CAFFE_DETAIL_FILLER_HPP
#define CAFFE_DETAIL_FILLER_HPP


#include <string>

#include "blob.hpp"
#include "proto/caffe.pb.h"
#include "syncedmem.hpp"
#include "util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler{
    public:
        explicit Filler(const FillerParameter& param) : filler_param_(param){}
        virtual ~Filler(){};
        virtual  void Fill(Blob<Dtype>*blob) = 0;

    protected:
        FillerParameter filler_param_;

    }; //class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
public:
    explicit  ConstantFiller(const FillerParameter & param) : Filler<Dtype>(param) {}


    //可以传入blob
    virtual void Fill(Blob<Dtype>* blob){
        Dtype * data = blob->mutable_cpu_data(); //将可读写cpu指针赋值给 data指针
        const int count = blob->count(); //取得传入的blob大小
        const Dtype value = this->filler_param_.value();

        CHECK(count);
        for (int i = 0; i < count; ++i) { //将value填进data指针
            data[i] = value;
        }
        CHECK_EQ(this->filler_param_.sparse(), -1)
            <<"Sparsity not supported by this Filler.";
        //constantFiller 不支持稀疏性
    }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
public:
    explicit UniformFiller(const FillerParameter & param) : Filler<Dtype>(param){}
    virtual void Fill(Blob<Dtype> * blob) {
        CHECK(blob->count());

        //caffe_rng_uniform 产生指定范围内的均匀分布随机数
        caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
                Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
        CHECK_EQ(this->filler_param_.sparse(), -1)
            <<"Sparsity not supported by this Filler.";
        //Uniform 不支持稀疏性
    }

};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
//高斯 支持sparse 稀疏性
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
public:
    explicit GaussianFiller(const FillerParameter &param) : Filler<Dtype>(param) {}

    virtual void Fill(Blob<Dtype> *blob) {
        Dtype *data = blob->mutable_cpu_data();
        CHECK(blob->count());
        // 调用caffe_rng_gaussian初始化、其中输入了高斯分布的均值和标准差
        caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_mean()),
                                  Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
        //GaussianFiller 支持稀疏性
        int sparse = this->filler_param_.sparse();
        CHECK_GE(sparse, -1); //确保sparse必须大于-1
        if (sparse >= 0) {
            // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
            // These have num == channels == 1; width is number of inputs; height is
            // number of outputs.  The 'sparse' variable specifies the mean number
            // of non-zero input weights for a given output.
            CHECK_GE(blob->num_axes(), 1);//确保blob维度大于1
            const int num_outputs = blob->shape(0);//调用shape(0) 返回shape_[0], NCHW中的N
            Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
            //非0的概率 = 1 / N

            rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
            int *mask = reinterpret_cast<int *>(rand_vec_->mutable_cpu_data());
            caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
            //caffe_rng_bernoulli 依照概率 分配0 or 1给 mask ????

            for (int i = 0; i < blob->count(); ++i) {
                data[i] *= mask[i];
            }
        }
    }

protected:
    shared_ptr<SyncedMemory> rand_vec_;
};
/*****************还没编写完*********************/
//    PositiveUnitballFiller
//    XavierFiller
//    MSRAFiller
//    BilinearFiller
/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter & param){
    const std::string & type = param.type();
    if (type == "constant"){
        return new ConstantFiller<Dtype>(param);
    } else if (type == "gaussian") {
        return new GaussianFiller<Dtype>(param);
    } else if (type == "positive_unitball") {
        return new PositiveUnitballFiller<Dtype>(param);
    } else if (type == "uniform") {
        return new UniformFiller<Dtype>(param);
    } else if (type == "xavier") {
        return new XavierFiller<Dtype>(param);
    } else if (type == "msra") {
        return new MSRAFiller<Dtype>(param);
    } else if (type == "bilinear") {
        return new BilinearFiller<Dtype>(param);
    } else {
        CHECK(false) << "Unknown filler name: " << param.type();
    }
        return (Filler<Dtype>*)(NULL);
}


};//namespace caffe closed
#endif //CAFFE_DETAIL_FILLER_HPP