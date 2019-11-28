//
// Created by StephenFang on 2019/11/28.
//

#ifndef CAFFE_DETAIL_DROPOUT_LAYER_HPP
#define CAFFE_DETAIL_DROPOUT_LAYER_HPP

#include <vector>
#include "../blob.hpp"
#include "../proto/caffe.pb.h"
#include "../layers/neuron_layer.hpp"

namespace caffe {
    /**
 * @brief During training only, sets a random portion of @f$x@f$ to 0, adjusting
 *        the rest of the vector magnitude accordingly.
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the inputs @f$ x @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the computed outputs @f$ y = |x| @f$
 */

template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {

public:
    /**
     * @param param provides DropoutParameter dropout_param,
     *     with DropoutLayer options:
     *   - dropout_ratio (\b optional, default 0.5).
     *     Sets the probability @f$ p @f$ that any given unit is dropped.
     */
    explicit DropoutLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {return "Dropout";}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /// when divided by UINT_MAX, the randomly generated values @f$u\sim U(0,1)@f$
    Blob<unsigned int> rand_vec_;
    /// the probability @f$ p @f$ of dropping any input
    Dtype threshold_;
    /// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
    Dtype scale_;
    //这里的scale 主要是Inverted Dropout实现方法， 在train的时候就先将dropout后的值乘1/p 来让结果的scale保持不变
    unsigned int uint_thres_;

};

}//namespace caffe closed

#endif //CAFFE_DETAIL_DROPOUT_LAYER_HPP
