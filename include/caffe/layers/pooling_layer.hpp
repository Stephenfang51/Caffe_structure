//
// Created by StephenFang on 2019/11/28.
//

#ifndef CAFFE_DETAIL_POOLING_LAYER_HPP
#define CAFFE_DETAIL_POOLING_LAYER_HPP

#include <vector>
#include "../blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class PoolingLayer : public Layer<Dtype>{
    explicit PoolingLayer(const LayerParameter& param):Layer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Pooling"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }

    //MAX POOL layers can output an extra top blob for the mask;
    //other can only output the pooled inputs
    virtual inline int MaxTopBlobs() const {
        return (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) ? 2: 1;
    }
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
    int pooled_height_, pooled_width_; //池化之后的宽高
    bool global_pooling_;
    PoolingParameter_RoundMode round_mode_;
    Blob<Dtype> rand_idx_;
    Blob<int> max_idx_;

};//class Pooling Layer closed
}//namespace caffe closed

#endif //CAFFE_DETAIL_POOLING_LAYER_HPP
