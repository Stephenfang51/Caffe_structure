//
// Created by StephenFang on 2019/11/22.
//

#include <vector>
#include <algorithm>
#include "../../../include/caffe/layers/batch_norm_layer.hpp"
#include "../../../include/caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>

/**参考https://www.cnblogs.com/LaplaceAkuir/p/7811351.html
    https://blog.csdn.net/mrhiuser/article/details/52575951**/

/**
 * @brief HW的归一化，求出NC个均值与方差，然后N个均值与方差求出一个均值与方差的Vector，
 * size为C，即相同通道的一个mini_batch的样本求出一个mean和variance
 * **/
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
    //获取BatchNormParameter参数列表
    BatchNormParameter param = this->layer_param_.batch_norm_param();
    //得到moving_average_fraction参数
    moving_average_fraction_ = param.moving_average_fraction();
    //指的是mini-batch时每次叠加mean的时候的衰退值

    use_global_stats_ = this->phase_ == TEST;

    if( param.has_use_global_stats())
        use_global_stats_ = param.use_global_stats();
    //如果num_axes = 1表示 channels_ 只有1
    if(bottom[0]->num_axes() == 1)
        channels_ = 1;

    else
        channels_ = bottom[0]->shape(1); //否则取第二维 得到channels_
    eps_ = param.eps();

    if (this->blobs_.size()>0) { //TEST的时候， 已经有值
        LOG(IFNO) << "Skipping parameter initialization"
        //保存mean variance, 系数
    }else {
        // 均值滑动，方差滑动，滑动系数
        this->blobs.resize(3);
        vector<int> sz;
        sz.push_back(channels_);
        this->blobs_[0].reset(new Blob<Dtype>(sz)); //C
        this->blobs_[1].reset(new Blob<Dtype>(sz)); //C

        sz[0] = 1;
        this->blobs_[2].reset(new Blob<Dtype>(sz)); //1
        for (int i = 0; i < 3; ++i){
            caffe_set(this->blobs_[i]->count(), Dtype(0),
                    this ->blobs_[i]->mutable_cpu_data());
            //blobs_[0~2]全部赋值为0
        }
    }
    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.

    for (int i = 0; i < this->blobs_.size(); ++i){
        if(this->layer_param_.param_size() == i){
            ParamSpec * fixed_param_spec = this->layer_param_.add_param();
            fixed_param_spec->set_lr_mult(0.f);
        }else {
            CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
                << "Cannot configure batch normalization statistics as layer "
                << "parameters.";
        }
    }
}

///Reshape,根据BN层在网络的位置，调整bottom和top的shape
//
//  *spatial_sum_multiplier_是一副图像大小的空间(height*width)，并初始化值为 1 ，
//  *作用是在计算mean_时辅助通过乘的方式将一副图像的值相加，结果是一个数值
//  */
template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
    if (bottom[0]->num_axes()>=1) //确认输入的Layer正常且channel一样
        CHECK_EQ(bottom[0]->shape(1), channels_);
    top[0]->ReshapeLike(*bottom[0]);

    vector<int> sz;
    sz.push_back(channels_);

    //mean_, variance_, temp_, x_norm_ 都是blob
    mean_.Reshape(sz);//通道数,即channel值大小，存储的是均值
    variance_.Reshape(sz);//通道数，即channel值大小，存储的是方差值
    temp_.ReshapeLike(*bottom[0]);//temp_中存储的是减去mean_后的每一个数的方差值。
    x_norm_.ReshapeLike(*bottom[0]);
    sz[0] = bottom[0] ->shape(0);
    batch_sum_multiplier_.Reshape(sz);

    int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
    //图像height*width

    if (spatial_sum_multiplier_.num_axes() == 0 ||
            spatial_sum_multiplier_.shape(0) != spatial_dim) {
        sz[0] = spatial_dim;
        spatial_sum_multiplier_.Reshape(sz);
        Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
        //分配一副图像的空间

        //初始化为1， 方便求和
        caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
    }
    // N*C,保存H*W后的结果,会在计算中结合data与spatial_dim求出
    int numbychans = channels_*bottom[0]->shape(0);
    if (num_by_chans_.num_axes() == 0 ||
        num_by_chans_.shape(0) != numbychans) {
        sz[0] = numbychans;
        num_by_chans_.Reshape(sz);
    //batch_sum_multiplier_ batch_size大小的空间，也是辅助在计算mean_时，将所要图像的相应的通道值相加。
        caffe_set(batch_sum_multiplier_.count(), Dtype(1),
                  batch_sum_multiplier_.mutable_cpu_data());
    }
}

/** BatchNorm 公式
// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
//
// dE(Y)/dX =
//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
//     ./ sqrt(var(X) + eps)
//
// where \cdot and ./ are hadamard product and elementwise division,
**/


template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0] ->shape(0);
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

    if (use_global_stats_) { //测试阶段使用全局均值
        //use th stored mean/variance estimates
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0?
                0:1 / this->blobs_[2]->cpu_date()[0];
        caffe_cpu_scale(variance_.count(), scale_factor,
                this->blbos_[0]->cpu_data(), mean_.mutable_cpu_data());
        //scale_factor * blobs_[0]->cpu_data()
        //y = alpha * x

    }else {

        //将每一副图像值相加为一个值，共有channels_ * num个值，然后再乘以 1/num*spatial_dim，结果存储到blob num_by_chans_中

        ///计算均值
        //训练阶段  compute mean
        //1.计算均值,先计算HW的均值，下一步在计算包含N个的均值
        // caffe_cpu_gemv 实现 y =  alpha*A*x+beta*y;
        // 输出的是channels_*num,
        //每次处理的列是spatial_dim，由于spatial_sum_multiplier_初始为1，即NCHW中的
        // H*W值相加，得到N*C*average个值，此处多除以了num，下一步可以不除以
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
                              1. / (num * spatial_dim), bottom_data,
                              spatial_sum_multiplier_.cpu_data(), 0.,
                              .mutable_cpu_data());
                //channels_ * num = A 行数
                //spatial_dim = A的列数
                //1. / (num * spatial_dim) = alpha 乘上 相当于 除以计算HW均值
                // bottom_data = A矩阵
                // spatial_sum_multiplier_.cpu_data() = x vector
                // 0. = beta
                //num_by_chans_.mutable_cpu_data() = y 计算出的N*C个均值存储到这边


        //2.计算均值，计算N各的平均值.
        // 由于输出的是channels个均值，因此需要转置
        // 上一步得到的N*C的均值，再按照num求均值，因为batch_sum全部为1,


        //上面计算得到的值大小是num*channel， 将图像的每个通道的值相加，最后获得channel个数值，结果存储到mean_中
        caffe_cpu_gemv<Dtype>(CBlasTrans,num,channels_,1,
                              num_by_chans_.cpu_data(),batch_sum_multiplier_.cpu_data(),
                              0,mean_.mutable_cpu_data());

        ///减去均值
        //batch_sum_multiplier_ : num x 1
        //mean_ : 1 x channels_
        //num_by_chans_ : num x channels_
        //num_by_chans_ :
        //         channels_
        // -----------------------
        // mean00 mean01 ... mean0x
        // ........................
        // ........................
        // meany0 meany1 ... meanyx
        // ------------------------
        //where x = channels and y = num

        caffe_cpu_gemm<Dtype>(CBlasNoTrans,CBlasNoTrans,num,channels_,1,1,
                              batch_sum_multiplier_.cpu_data(),mean_.cpu_data(),0,
                              num_by_chans_.mutable_cpu_data());
        //num_by_chans_: (channels_ * num) x 1
        //spatial_sum_multiplier_ : 1 x spatial_dim (all values are 1)
        //top_data = 1 x top_data + (-1) x (num_by_chans_ x spatial_sum_multiplier_)
        //这里num_by_chans_ x spatial_sum_multiplier_求得的是每个值对应的平均值
        //最后的top_data保存的就是每个值减去对应channel的均值后的结果

        caffe_cpu_gemm<Dtype>(CBlasNoTrans,CBlasNoTrans,num*channels_,
                              spatial_dim,1,-1,num_by_chans_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(),1, top_data());

        ///计算方差 compute variance using var(X) = E((X-EX)^2)
        if (!use_global_stats_) {
            // compute variance using var(X) = E((X-EX)^2)
            caffe_sqr<Dtype>(top[0]->count(), top_data, //计算方差将结果保存于temp_中
                             temp_.mutable_cpu_data());  // (X-EX)^2

            //这步计算的是对应每个channel的(X-EX)^2的和，同时除以1. / (num * spatial_dim)
            //这步和计算mean的时候很相似
            caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
                                  1. / (num * spatial_dim), temp_.cpu_data(),
                                  spatial_sum_multiplier_.cpu_data(), 0.,
                                  num_by_chans_.mutable_cpu_data());
            //这步计算的是对应每个batch的(X-EX)^2的和，此时不需要再除以num了，因为上一步已经除以了
            //这步和计算mean的时候很相似
            //这样就得到variance了
            caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                                  num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                                  variance_.mutable_cpu_data());  // E((X_EX)^2)

            ///计算并且保存滑动系数
            // blobs_[2]中只有一个值，最开始值为0
            // 然后每次blobs_[2] = （moving_average_fraction_ * blobs_[2]）+ 1
            this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
            this->blobs_[2]->mutable_cpu_data()[0] += 1;
            //caffe_cpu_axpby : Y = alpha * X + b * Y
            //blobs_[0] = 1 * mean_ + moving_average_fraction_ * blobs_[0]
            //最开始blobs_[0]中的所有值为0
            //这一步是用来保存并叠加每一次计算得到的mean值，
            //结果保存在blob_[0]中

            caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
                            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
            int m = bottom[0]->count() / channels_;
            Dtype bias_correction_factor = m > 1 ? Dtype(m) / (m - 1) : 1;
            caffe_cpu_axpby(variance_.count(), bias_correction_factor,
                            variance_.cpu_data(), moving_average_fraction_,
                            this->blobs_[1]->mutable_cpu_data());
        }
        // normalize variance
        caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
        caffe_sqrt(variance_.count(), variance_.cpu_data(),
                   variance_.mutable_cpu_data());

        // replicate variance to input size
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                              batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
                              num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                              spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
        caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
        // TODO(cdoersch): The caching is only needed because later in-place layers
        //                 might clobber the data.  Can we skip this if they won't?
        caffe_copy(x_norm_.count(), top_data,
                   x_norm_.mutable_cpu_data());
    }

}//forward closed

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff;
    if (bottom[0] != top[0]) {
        top_diff = top[0]->cpu_diff();
    } else {
        caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
        top_diff = x_norm_.cpu_diff();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (use_global_stats_) {
        caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
        return;
    }
    const Dtype* top_data = x_norm_.cpu_data();
    int num = bottom[0]->shape()[0];
    int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,
    // respectively, dE/dY is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.

    // sum(dE/dY \cdot Y)
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
                          bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
                          num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                          num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                          mean_.mutable_cpu_data());

    // reshape (broadcast) the above
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                          num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                          spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                          spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
                          top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
                          num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
                          num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
                          mean_.mutable_cpu_data());
    // reshape (broadcast) the above to make
    // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                          batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                          num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
                          spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                          spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
                    Dtype(-1. / (num * spatial_dim)), bottom_diff);

    // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}
}//namespace caffe closed
