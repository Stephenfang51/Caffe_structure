//
// Created by StephenFang on 2019/11/28.
//

#include <vector>
#include "../../../include/caffe/layers/flatten_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        CHECK_NE(top[0], bottom[0]) << this->type() << "Layer does not "
                                                       "allow in-place computation.";

        //获取参数起始维度以及结束维度
        const int start_axis = bottom[0] ->CanonicalAxisIndex(
                this->layer_param_.flatten_param().axis());
        const int end_axis = bottom[0]->CanonicalAxisIndex(
                this ->layer_param_.flatten_param().end_axis());


        //假设上层传入的为(1, 3, 28, 28)
        //top_shape 先行建立一个记录输出尺寸的vector
        //然后第一步找出第一个维度， 也就是没有被指定要flattened的， 通常是N =>top_shape =(1,)
        //创建一个记录flattened_dim， 也就是 3* 28 * 28
        //top_shape 在将刚刚计算出来的flattened_dim 加入， => (1, 2352)
        //

        vector<int> top_shape;

        for (int i = 0; i < start_axis; ++i){
            top_shape.push_back(bottom[0]->shape(i));
        }


        const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
        //指定flat后的维度数


        top_shape.push_back(flattened_dim);
        for (int i = end_axis + 1; i < bottom[0] ->num_axes(); ++i){
            top_shape.push_back(bottom[0]->shape(i));
        }
        top[0]->Reshape(top_shape); //将记录好的shape, 赋值输出top
        CHECK_EQ(top[0]->count(), bottom[0]->count());
        //检查输入与输出的总维度数是相等的才可以
    }

template <typename Dtype>
void FlattenLayer<Dtype>::Forward_cpu(const vector<caffe::Blob<Dtype> *> &bottom,
                                      const vector<caffe::Blob<Dtype> *> &top) {
    top[0]->ShareData(*bottom[0]); //将bottom的值赋值给top
}

template <typename Dtype>
void FlattenLayer<Dtype>::Backward_cpu(const vector<caffe::Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                       const vector<caffe::Blob<Dtype> *> &bottom) {
    bottom[0]->ShareDiff(*top[0]);
}



}//namespace caffe closed


