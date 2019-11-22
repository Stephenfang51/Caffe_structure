//
// Created by StephenFang on 2019/11/21.
//

#ifndef CAFFE_DETAIL_SOFTMAX_LAYER_HPP
#define CAFFE_DETAIL_SOFTMAX_LAYER_HPP

#include <vector>
#include "../blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {
    /**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
    explicit SoftmaxLayer(const LayerParameter & param) : Layer<Dtype>(param) {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {return "Softmax";}
    virtual inline int ExactNumBottomBlobs() const { return 1;}
    virtual inline int ExactNumToplobs() const { return 1;}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int outer_num_; //batch_size ?
    int inner_num_; //height *  width
    int softmax_axis_;

    /// sum_multiplier is used to carry out sum using BLAS
    Blob<Dtype> sum_multiplier_;
    /// scale is an intermediate Blob to hold temporary results.
    Blob<Dtype> scale_;
};
}//namespace caffe closed

#endif //CAFFE_DETAIL_SOFTMAX_LAYER_HPP
