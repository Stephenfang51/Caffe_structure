//
// Created by StephenFang on 2019/11/21.
//

#include <algorithm>
#include <vector>
#include "../../../include/caffe/layers/softmax_layer.hpp"
#include "../../../include/caffe/util/math_functions.hpp"


/**参考https://blog.csdn.net/sinat_22336563/article/details/70144228**/
namespace caffe{
template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                  const vector<Blob<Dtype>*> & top) {
    //取得softmax
    //经过全连接后，现在输入为input_num*output_num的矩阵




    softmax_axis_ = //默认为1
            bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

    top[0]->ReshapeLike(* bottom[0]);//input_num * output_num

    vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));//output_num
    sum_multiplier_.Reshape(mult_dims); //output_num
    Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();//outer_num_ =input_num
    caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);//inner_num_=count = 1;
    //  赋值Dtype(1) for multiplier_data

    outer_num_ = bottom[0]->count(0, softmax_axis_); //outer_num_ =input_num
    inner_num_ = bottom[0]->count(softmax_axis_ + 1);//inner_num_=count = 1;

    vector<int> scale_dims = bottom[0]->shape();
    scale_dims[softmax_axis_] = 1;
    scale_.Reshape(scale_dims);//最后结果为64*1的向量

}
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    const Dtype * bottom_data = bottom[0]->cpu_data();
    //取得 l 层 可写data
    Dtype * top_data = top[0] -> mutable_cpu_data();
    Dtype * scale_data = scale_.mutable_cpu_data();
    // scale is an intermediate Blob to hold temporary results.


    int channels = bottom[0]->shape(softmax_axis_);
    int dim = bottom[0]->count() / outer_num_; 
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    //We need to subtract the max to avoid numerical issues, compute the exp,
    //and then normalize.

    /** softmax步骤

     (1）找出输入的最大值；

    （2）输入的每一个变量都减去最大值；

    （3）对（2）中结果求去指数函数； top_data

    （4）对（3）中结果归一化，得出的结果就是输入在每一个标签上概率  top_data(y)/scale_data(sum)
     **/


    for (int i=0; i < outer_num_;++i ){
        //初始化scale_data to the first plane
        caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
        //bottom_data + i * dim 取代scale_data值
        //inner_num_ = H*W
        //将输入复制到sacle_data中缓存

        for (int j = 0; j< channels; j++){ {//针对每一个类别
            for (int k = 0; k< inner_num_; k++){//针对每一张输入的图片，按照每个像素点进行计算
                scale_data[k] = std::max(scale_data[k],
                        bottom_data[i * dim + j * inner_num_ + k]);
                //找出输入Zi的最大值， 并且保存于scale_data
            }
        }
        ///subtraction 减去最大值
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1, -1,
                sum_multiplier_.cpu_data(), scale_data, 1., top_data);
        //公式为 C=alpha*A*B+beta*C， alpha 值为-1，
        //A矩阵为sum_multiplier_,  B矩阵为scale_data



        ///exponentitation exp
        caffe_exp<Dtype>(dim, top_data, top_data); //argument n, a, y : y[i] = exp(a[i])
        ///sum after exp 将所有exp结果加总之后， 用来Normalize output的， 参考softmax公式
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1., top_data,
                sum_multiplier_.cpu_data(), 0., scale_data);
        //将元素变为1， 矩阵A为top_data, A的M = channels, N为 inner_num,
        //vector x = sum_multiplier_.cpu_data()
        //vector y = scale_data
        //alpha * A(trnasposed)* X + beta * Y

        ///division ：每个exp之后的值除以sum（scale_data)值求出y， normalized
        for(int j=0; j < channels; j++) {
            caffe_div(inner_num_, top_data, scale_data, top_data);//vdDiv(n, a, b, y); //y  = a/b 按元素
            top_data += inner_num_;//地址移动到下一个图片，此处为+1
        }
    }
}
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {


    const Dtype*top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();

    Dtype* bottom_diff = bottom[0] ->mutalble_cpu_diff();
    Dtype* scale_data = scale_.mutable_cpu_diff();
    int channels = top[0] ->shape(softmax_axis_);
    int dim = top[0]->count() / outer_num_;
    caffe_copy(top[0]->count(), top_diff, bottom_diff);

    for(int i = 0; i < outer_num; ++i){ //outer_num是批次， 预测时一般为1
        //  计算top_diff和top_data的点积，然后从bottom_diff中减去该值  top_diff*top_data

        for ( int k=0; k< inner_num; ++k){
            scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
                bottom_diff + i * dim + k, inner_num_,
                top_data + i * dim + k, inner_num_);
        }
        //  bottom_diff=top_diff-top_diff*top_data
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
            -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
    }
    // top_data*(top_diff-top_diff*top_data)
    caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);

}

}//namespace caffe closed