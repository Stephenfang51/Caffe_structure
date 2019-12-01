//
// Created by StephenFang on 2019/11/29.
//
#include <algorithm>
#include <cfloat>
#include <vector>

#include "../../../include/caffe/layers/pooling_layer.hpp"
#include "../../../include/caffe/util/math_functions.hpp"
#include "../../../include/caffe/proto/caffe.pb.h"

namespace caffe {
using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<caffe::Blob<Dtype> *> &bottom,
                                     const vector<caffe::Blob<Dtype> *> &top) {
    //获取参数
    PoolingParameter pool_param = this->layer_param_.pooling_param();

    //如果global_pooling = True
    if (pool_param.global_pooling()) {
        CHECK((!pool_param.has_kernel_size() || pool_param.has_kernel_h()
        || pool_param.has_kernel_w()))
        << "With Global_pooling: true Filter size cannot specified";

    }else {
        CHECK(!pool_param.has_kernel_size() !=
              !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
              <<"Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(pool_param.has_kernel_size() ||
                      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
              <<"For non-square filters both kernel_h and kernel_w are required.";
    }
    CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
           && pool_param.has_pad_w())
          || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
                    << "pad is pad OR pad_h and pad_w are required.";
    CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
           && pool_param.has_stride_w())
          || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
                    << "Stride is stride OR stride_h and stride_w are required.";

    global_pooling_ = pool_param.global_pooling(); //bool
    round_mode_ = pool_param.round_mode();

    //如果global_pooling_ = true，
    if ( global_pooling_){
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    } else {
        if(pool_param.has_kernel_size()){
            kernel_h_ = kernel_w_ = pool_param.kernel_size();
        }else {
            kernel_h_ = pool_param.kernel_h();
            kernel_w_ = pool_param.kernel_w();
        }

    }
    //确保卷积核size必须大于0
    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";


    //获取pad参数
    if (!pool_param.has_pad_h()){
        pad_h_ = pad_w_ = pool_param.pad();
    } else{
        pad_h_ = pool_param.pad_h();
        pad_w_ = pool_param.pad_w();
    }

    //获取步长参数
    if (!pool_param.has_stride_h()){
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_h();
    }

    //如果是全局池化， 检查pad 是否为0, stride_h 为1
    if (global_pooling_){
        CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
        <<"With Global_pooling: true; only pad = 0 and stride =1";
    }

    //确认设定了AVE 或者 MAX
    if(pad_h_ != 0 || pad_w_ != 0){
        CHECK(this->layer_param_.pooling_param().pool()
              == PoolingParameter_PoolMethod_AVE
              || this->layer_param_.pooling_param().pool()
              == PoolingParameter_PoolMethod_MAX)
              << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_h_, kernel_h_); //确定pad值不能比kernel值还大
        CHECK_LT(pad_w_, kernel_w_);
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){

    //确保输入数据为4维度
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                          <<" correspoding to (num, channels, height, width)";
    //获取输入的数据
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    if (global_pooling_){
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    }
    switch(round_mode_) {


        // math.ceil 表示向上取正+1， floor向下， 小数都直接舍弃
        // 求出池化采样之后的特征图大小
        case PoolingParameter_RoundMode_CEIL:

            pooled_height_ = static_cast<int>(ceil(static_case<float>(
                    height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
            pooled_width_ = static_cast<int> (ceil(static_case<float>(
                    width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
            break;

        case PoolingParameter_RoundMode_FLOOR:
            pooled_height_ = static_cast<int>(floor(static_case<float>(
                    height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
            pooled_width_ = static_cast<int> (floor(static_case<float>(
                    width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
            break;

        default:
            LOG(FATAL) << "Unknown rouding mode. ";
    }
    if (pad_h_ || pad_w_){ //如果有padding
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        // 如果有图像补齐，则需要确保不发生越界，否则不做最后一个采样点
        if((pooled_height_ -1) * stride_h_ >= height_ + pad_h_){
            --pooled_height_;
        }
        if((pooled_width_ -1) * stride_w_ >= width_ + pad_w_){
            --pooled_width_;
        }
        CHECK_LT((pooled_height_ - 1)* stride_w_, height_ + pad_w_);
        CHECK_LT((pooled_width_ - 1)* stride_w_, width_ + pad_w_);

    }

    //将输出数据top 依照计算好的pooled height width改变
    top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    if (top.size()> 1){
        top[1]->ReshapeLike(*top[0]);
    }

    //if max pooling, we will initialize the vector index part.
    //如果是max pooling, 初始化max pooling 索引？？？
    if (this->layer_param_.pooling_param().pool() ==
        PoolingParameter_PoolMethod_MAX && top.size() == 1 ){
        max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    }
    //If stochastic pooling, we will initialize the random index part
    if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_STOCHASTIC){
        rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    //赋值输入与输出
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype * top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();


    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    //当输出的超过一个元素 就使用top_mask

    int* mask = NULL; // suppress warnings about uninitialized variables
    Dtype* top_mask = NULL;
    //Different pooling methods, We explicity do the switch outside the for
    //loop to save time, although this results in more code.


    ///判别pooling 方式 MAX or AVE，赋值-1 给top_mask or mask
    switch (this -> layer_param_.pooling_param().pool()) {
        case PoolingParameter_PoolMethod_MAX:
            //初始化
            if (use_top_mask) { //top 输出大于1个元素的时候， 用 top_mask
                top_mask = top[1]->mutable_cpu_data();
                caffe_set(top_count, Dtype(-1), top_mask); //赋值-1给top_mask
            } else { //top 输出不超过1个元素的时候， 用 mask
                mask = max_idx_.mutable_cpu_data();
                caffe_set(top_count, -1, mask); //mask => int type
            }
            caffe_set(top_count, Dtype(-FLT_MAX), top_data); // FLT_MAX = 3.402823466e+38F
            //The main loop

            //遍历取最大值
            for (int n = 0; n < bottom[0]->num(); ++n) {
                for (int c = 0; c < channels_; ++c) {
                    //ph, pw 池化范围长宽， < pooled_hegiht/width 不超过范围
                    for (int ph = 0; ph < pooled_height_; ++ph) {
                        for (int pw = 0; pw < pooled_width_; ++pw) {

                            //防止kernel 滑动到图像范围外， 计算HW每一次的起始点与尾巴
                            int hstart = ph * stride_h_ - pad_h_;
                            int wstart = pw * stride_w_ - pad_w_;
                            int hend = min(hstart + kernel_h_, height_);
                            int wend = min(wstart + kernel_w_, width_);
                            hstart = max(hstart, 0); //表示不得小于0
                            wstart = max(wstart, 0);

                            //输出图像上， 也就是池化后像素点的位置
                            const int pool_index = ph * pooled_width_ + pw;

                            //遍历不超过hend 以及 wend
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {

                                    //目前输入图像上像素的位置
                                    const int index = h * width_ + w;

                                    //如果输出的值较输入小， 则输出变成输入的值， 也就是取大的
                                    //思路就是原输入图像上的像素点一个一个遍历去和输出（池化后）的像素点比大小， 然后取大(max)的值
                                    if (bottom_data[index] > top_data[pool_index]) {
                                        top_data[pool_index] = bottom_data[index];

                                        //如果使用top_mask
                                        if (use_top_mask) {
                                            top_mask[pool_index] = static_cast<Dtype>(index);
                                        } else {
                                            mask[pool_index] = index;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    //compute offset 处理完一个channel之后 换 下一个channel, 然后重新遍历下一个图像
                    //offset(n, c, h, w)
                    bottom_data += bottom[0]->offset(0, 1);
                    top_data += top[0]->offset(0, 1);
                    if (use_top_mask) {
                        top_mask += top[0]->offset(0, 1);
                    } else {
                        mask += top[0]->offset(0, 1);
                    }
                }
            }
            break;
        case PoolingParameter_PoolMethod_AVE:
            for (int i = 0; i < top_count; ++i) {
                top_data[i] = 0;
            }
            //The main loop
            for (int n = 0; n < bottom[0]->num(); ++n) {
                for (int c = 0; c < channels_; ++c) {
                    for (int ph = 0; ph < pooled_height_; ++ph) {
                        for (int pw = 0; pw < pooled_width_; ++pw) {
                            int hstart = ph * stride_h_ - pad_h_;
                            int wstart = pw * stride_w_ - pad_w_;
                            int hend = min(hstart + kernel_h_, height_ + pad_h_);
                            int wend = min(wstart + kernel_w_, width_ + pad_w_);


                            int pool_size = (hend - hstart) * (wend - wstart);
                            hstart = max(hstart, 0);
                            wstart = max(wstart, 0);
                            hend = min(hend, height_);
                            wend = min(wend, width_);

                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    top_data[ph * pooled_width_ + pw] +=
                                            bottom_data[h * width_ + w];
                                }
                            }
                            top_data[ph * pooled_width_ + pw] /= pool_size; //做平均
                        }
                    }
                    //compute offset
                    bottom_data += bottom[0]->offset(0, 1);
                    top_data += top[0]->offset(0, 1);

                }


            }
            break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
            NOT_IMPLEMENTED;
            break;
        default:
            LOG(FATAL) << "Unknown pooling method.";
        }
    }

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<caffe::Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                       const vector<caffe::Blob<Dtype> *> &bottom) {
    if (!propagate_down[0]){
        return;
    }
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Different pooling methods. We explicitly do the switch outside the for
    // loop to save time, although this results in more codes.
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff); //0赋值
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size()> 1;
    const int* mask = NULL; //suppress warnings about uninitialized variables
    const Dtype* top_mask = NULL;
    switch (this->layer_param_.pooling_param().pool()){
        case PoolingParameter_PoolMethod_MAX:
            //The main loop
            if (use_top_mask){
                top_mask = top[1]->cpu_data();
            }else {
                mask = max_idx_.cpu_data();
            }
            for ( int n = 0; n < top[0]->num(); ++n){
                for (int c = 0; c< channels_; ++c){
                    for (int ph = 0; ph<pooled_height_; ++ph){
                        for (int pw=0; pw<pooled_width_; ++pw){
                            const int index = ph * pooled_width_ + pw;
                            const int bottom_index =
                                    use_top_mask ? top_mask[index] : mask[index];
                            bottom_diff[bottom_index] += top_diff[index];
                        }
                    }

                    //与forward一样 偏移量增加
                    bottom_diff += bottom[0]->offset(0, 1);
                    top_diff += top[0]->offset(0, 1);
                    if(use_top_mask){
                        top_mask += top[0]->offset(0, 1);
                    }else {
                        mask += top[0]->offset(0, 1);
                    }
                }
            }
            break;
        case PoolingParameter_PoolMethod_AVE:
            // The main loop
            for (int n = 0; n < top[0]->num(); ++n) {
                for (int c = 0; c < channels_; ++c) {
                    for (int ph = 0; ph < pooled_height_; ++ph) {
                        for (int pw = 0; pw < pooled_width_; ++pw) {
                            int hstart = ph * stride_h_ - pad_h_;
                            int wstart = pw * stride_w_ - pad_w_;
                            int hend = min(hstart + kernel_h_, height_ + pad_h_);
                            int wend = min(wstart + kernel_w_, width_ + pad_w_);
                            int pool_size = (hend - hstart) * (wend - wstart);
                            hstart = max(hstart, 0);
                            wstart = max(wstart, 0);
                            hend = min(hend, height_);
                            wend = min(wend, width_);
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    bottom_diff[h * width_ + w] +=
                                            top_diff[ph * pooled_width_ + pw] / pool_size;
                                }
                            }
                        }
                    }
                    // offset
                    bottom_diff += bottom[0]->offset(0, 1);
                    top_diff += top[0]->offset(0, 1);
                }
            }
            break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
            NOT_IMPLEMENTED;
            break;
        default:
            LOG(FATAL) << "Unknown pooling method.";
    }
}


#ifdef CPU_ONLY
    STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);
}//namespace caffe closed