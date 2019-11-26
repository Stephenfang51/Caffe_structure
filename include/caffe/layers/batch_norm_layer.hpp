//
// Created by StephenFang on 2019/11/22.
//

#ifndef CAFFE_DETAIL_BATCH_NORM_LAYER_HPP
#define CAFFE_DETAIL_BATCH_NORM_LAYER_HPP

#include <vector>
#include "../blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {

    /**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */


template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
public:
    explicit BatchNormLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "BatchNorm"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


    Blob<Dtype> mean_, variance_, temp_, x_norm_;
    bool use_global_stats_;
    Dtype moving_average_fraction_; //指的是mini-batch时每次叠加mean的时候的衰退值
    int channels_;
    Dtype eps_; // 防止分母为0

    // extra temporarary variables is used to carry out sums/broadcasting
    // using BLAS

    // 中间变量，理解了BN的具体过程即可明了为什么需要这些
    Blob<Dtype> batch_sum_multiplier_; // 长度为N*1，全为1，用以求和， 用处和spatial_sum_multiplier差不多
    Blob<Dtype> num_by_chans_; // 临时保存H*W的结果，长度为N*C
    Blob<Dtype> spatial_sum_multiplier_;
    // 用来相乘， matrix * vector 执行inner product用， 通常spatial_sum_multiplier的元素值为1
    // 然后特征图上的数值就相当于加总起来
};

}//namespace caffe closed

#endif //CAFFE_DETAIL_BATCH_NORM_LAYER_HPP
