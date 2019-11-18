#include <algorithm>
#include <vector>
#include "../../../include/caffe/filler.hpp"
#include "../../../include/caffe/util/im2col.hpp"
#include "../../../include/caffe/util/math_functions.hpp"
#include "../../../include/caffe/layers/base_conv_layer.hpp"

/**  axis 和 spatial的概念
 * // The axis to interpret as "channels" when performing convolution.
// Preceding dimensions are treated as independent inputs;
// succeeding dimensions are treated as "spatial".
// With (N, C, H, W) inputs, and axis == 1 (the default), we perform
// N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
// groups g>1) filters across the spatial axes (H, W) of the input.
// With (N, C, D, H, W) inputs, and axis == 1, we perform
// N independent 3D convolutions, sliding (C/g)-channels
// filters across the spatial axes (D, H, W) of the input.
 */
namespace caffe {
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    //计算出kernel size, padding, stride and inputs
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    force_nd_im2col_ = conv_param.force_nd_im2col(); //force_nd_im2col 是 类方法
    channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis())//blob 调用CanonicalAxisIndex(), 负变为正
    const int first_spatial_axis = channel_axis_ + 1; //定义第一个空间(H, W)轴， 也就是height的index？
    const int num_axes = bottom[0]->num_axes();//blob 调用num_axes返回维度
    num_spatial_axes_ = num_axes - first_spatial_axis; //计算出空间轴的个数 也就是HW 2D应该为2个
    CHECK_GE(num_spatial_axes_, 0); //确定上一行代码计算出的大于等于0
    vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
    //vector 存放1个元素， 初始值为std::max(num_spatial_axes_, 1)
    /*****************设置kernel 维度 (kernel_shape_)************************/
    kernel_shape_.Reshape(spatial_dim_blob_shape);//设定好卷积核的维度, 记住kernel_shape_是blob类型
    int *kernel_shape_data = kernel_shape_.mutable_cpu_data();//调用可读写存在cpu上数据
    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
        //确认有卷积核的H， W

        CHECK_EQ(num_spatial_axes_, 2)
            << "kernel_h & kernel_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.kernel_size_size())
            << "Either kernel_size or kernel_h/w should be specified; not both.";
        kernel_shape_data[0] = conv_param.kernel_h(); //设置卷积核的H， W
        kernel_shape_data[1] = conv_param.kernel_h();
    } else {
        //否则的话， 从proto中取得kernel的尺寸
        //下面看不懂

        const int num_kernel_dims = conv_param.kernel_size_size();
        CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
                        << "kernel_sie must be specified once, or once per spatial dimension "
                        << " (kernel_size specified " << num_kernel_dims << " times; "
                        << num_spatial_axes_ << " spatial dims).";

        for (int i = 0; i < num_spatial_axes_; ++i) {
            //根据num_spatial_axes 的总数 赋值给kernel_shape_data[i]
            kernel_shape_data[i] =
                    conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
            //上面看不懂
        }
    }

    for (int i = 0; i < num_spatial_axes_; ++i) {
        //确认kernel的元素， 每一个 val1 大于  val2, 不能为0
        CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
    }

    /**************设置好 步长维度 (stride_)*****************/
    stride_.Reshape(spatial_dim_blob_shape); //注意stride_为blob类型，依照spatial_dim_blob_shape设置

    int *stride_data = stride_.mutable_cpu_data();//stride_data 指针 指向blob调用的可读写cpu数据

    if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
            << "stride_h & stride_w can only be used for 2D covolution.";
        CHECK_EQ(0, conv_param.stride_size())
            << "Either stride or stride_h/w should be specified; not both";
        stride_data[0] = conv_param.stride_h();
        stride_data[1] = conv_param.stride_w();
    } else {
        const int num_stride_dims = conv_param.stride_size();
        CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
              num_stride_dims == num_spatial_axes_)
                        << "stride must be specified once, or once per spatial dimension "
                        << "(stride specified " << num_stride_dims << " times; "
                        << num_spatial_axes_ << " spatial dims).";

        const int kDefaultStride = 1; // 默认步长
        for (int i = 0; i < num_spatial_axes_; ++i) { //依照空间轴数赋值
            stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                             conv_param.stride((num_stride_dims == 1) ? 0 : i);
            CHECK_GT(stride_data[i], 0) << "Stride demensions must be nonzero.";
        }
    }

    /*****************设置pad 维度************************/
    pad_.Reshape(spatial_dim_blob_shape);
    int *pad_data = pad_.mutable_cpu_data();
    if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
            << "pad_h & pad_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.pad_size())
            << "Either pad or pad_h/w should be specified; not both.";
        pad_data[0] = conv_param.pad_h();
        pad_data[1] = conv_param.pad_w();
    } else {
        const int num_pad_dims = conv_param.pad_size();
        CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
              num_pad_dims == num_spatial_axes_)
                        << "pad must be specified once, or once per spatial dimension "
                        << "(pad specified " << num_pad_dims << " times; "
                        << num_spatial_axes_ << " spatial dims).";
        const int kDefaultPad = 0;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                          conv_param.pad((num_pad_dims == 1) ? 0 : i);
        }
    }
    /**Setup dilation dimensions (dilation_).**/
    dilation_.Reshape(spatial_dim_blob_shape);
    int *dilation_data = dilation_.mutable_cpu_data();
    const int num_dilation_dims = conv_param.dilation_size();
    CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
          num_dilation_dims == num_spatial_axes_)
                    << "dilation must be specified once, or once per spatial dimension "
                    << "(dilation specified " << num_dilation_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
    const int kDefaultDilation = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                           conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
    }
    /**
     * Special case: im2col is the identity for 1x1 convolution with stride 1
     * and no padding, so flag for skipping the buffer and transformation.
     **/
    is_1x1_ = true;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        is_1x1_ &= //&= 按位“与”运算
                kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
        if (!is_1x1_) { break; }
    }
    /**计算output channel and groups**/
    channels_ = bottom[0]->shape(channel_axis_); //blob调用shape方法， 带入index取得相应的值
    num_output_ = this->layer_param_.convolution_param().num_output();
    //class layer调用protected 成员layer_param_ 并调用
    //proto中LayerParameter的方法 convolution_param()取得卷积层的参数并取得卷积核的个数

    CHECK_GT(num_output_, 0); //卷积核个数大于0

    //group的部分
    group_ = this->layer_param_.convolution_param().group();
    CHECK_EQ(channels_ % group_, 0); //意味着必须channels_除group_整除
    CHECK_EQ(num_output_ % group_, 0)  //意味着必须num_output_除group_整除
                << "Number of output should be multiples of group.";

    //判断是否为反卷积， 并作出相应的赋值
    if (reverse_dimensions()) {
        conv_out_channels_ = channels_;
        conv_in_channels_ = num_output_;
    } else {
        conv_out_channels_ = num_output_;
        conv_in_channels_ = channels_;
    }

    /******************处理参数 ： weight 及 bias 偏值*************/
    // blobs_[0] 对应 卷积核上的weights
    // blobs_[1] 对应 偏值（可选）

    vector<int> weight_shape(2); //创建一个vector容器， 大小为2
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
        weight_shape.push_back(kernel_shape_data[i]);
        //kernel_shape_data是int指针， 将kernel shape存放到weight_shape后面
    }

    bias_term_ = this->layer_param_.convolution_param().bias_term();
    vector<int> bias_shape(bias_term_, num_output_);
    //创建一个bias_shape容器， 大小为bias_term_ 非0即1


    //blobs_ 类型为vector 存储shared_ptr<Blob<Dtype>> from layer.hpp
    if (this->blobs_.size() > 0) { //调用private成员用this
        CHECK_EQ(1 + bias_term_, this->blobs_.size())
            << "Incorrect number of weight blobs.";
        //这一步是要确保权重weights的数量必须一样

        //blobs_为shared_ptr<Blob<Dtype>> 调用blob的函数 shape() 以及 shape_string()
        //shape返回数据的形状
        if (weight_shape != this->blobs_[0]->shape()) {
            //如果检查发现weight 和blobs_的形状不同， 创建一个与weight_shape相同的blob数据
            Blob<Dtype> weight_shaped_blob(weight_shape);
            LOG(FATAL) << "Incorrect weight shape: expeectd shape "
                       << weight_shaped_blob.shape_string() << "； instead， shape was "
                       << this->blobs_[0]->shape_string();
        }
        if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
            Blob<Dtype> bias_shaped_blob(bias_shape);
            LOG(FATAL) << "Incorrect bias shape: expected shape "
                       << bias_shaped_blob.shape_string() << "; insted, shape was "
                       << this->blobs_[1]->shape_string();
        }
        LOG(INFO) << "Skipping parameter initialization";
    } else { //如果blobs_.size 没有大于0 也就是没有东西， 则将blobs_重新resize
        if (bias_term_) { //如果有启用bias的话
            this->blobs_.resize(2);

        } else {
            this - blobs.resize(1); //没有bias 则只给weight, resize成1个
        }

        /**初始化以及填入weights**/
        // output channels x input channels per-group x kernel height x kernel width
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));


        //创建weight_filler智能指针
        //shared_ptr<> p1() 并且初始化 调用GetFiller函数
        shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
                this->layer_param_.convolution_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
        //weight_filler调用Fill， blobs_[0] 表示weight,

        //如果有bias ， 初始化以及填入biases
        if (bias_term_) {
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

            //创建bias_filler智能指针
            //shared_ptr<> p1() 并且初始化 调用GetFiller函数
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.convolution_param().bias_filler()));
            bias_filler->Fill(this->blobs_[1].get());
            //blobs_[1]表示bias

        }
    }
    kernel_dim_ = this->blobs_[0]->count(1);//调用blob中的count可以计算出从index1到最后的维度相乘
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    //kernel_dim_ = A的列与B的行
    //conv_out_channels = 输出图像的通道数
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}



/*** Reshape 重写 ***/
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*> & top) {
    const int first_spatial_axis = channel_axis_ +1;


    //num_axes 返回blob是多少维度， 确认维度必须相同
    CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
        << "bottom num_axes may not changed.";
    num_ = bottom[0]->count(0, channel_axis_); //统计输入特征图数量batch_size*channel_num
    CHECK_EQ(bottom[0] -> shape(channel_axis_), channels_)
    //确认传入的bottom 的通道数 与  channels_的通道数一致
        <<"Inut size incompatbile with convolution kernel.";

    // TODO: generalize to handle inputs of different shapes.

    // 检查所有bottom blob形状一致
    for(int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id){
        CHECK(bottom[0] ->shape() == bottom[bottom_id] -> shape())
            << "shape mismatch - bottom[0]: " << bottom[0] ->shape_string()
            <<" vs. bottom[" << bottom_id << "}:"
            << bottom[bottom_id] ->shape_string();
    }

    // Shape the tops.根据bottom形状，计算输出top blob形状（batch_size, channel_out, out_h, out_w,...）
    bottom_shape_ = &bottom[0]->shape();
    compute_output_shape(); //虚函数， 需要重写来确定具体形状

    //复制[begin,end)区间内另一个数组（描述bottom形状）的元素到该vector中，左闭右开，如果是blob是b*c*h*w，就只复制过来了b
    vector<int> top_shape(bottom[0]->shape().begin(),
            bottom[0]->shape().begin() + channel_axis_);
    top_shape.push_back(num_output_);//num_output_添加在后



    for (int i = 0; i < num_spatial_axes_; ++i) {
        top_shape.push_back(output_shape_[i]);
        //output_shape是输出的H, W
    }

    // 按得到的top_shape创建top blob，并调整其形状，为其开辟空间等。
    for (int top_id = 0; top_id <top.size(); ++top_id) {
        top[top_id] ->Reshape(top_shape);
    }
    // 按得到的top_shape创建top blob，并调整其形状，为其开辟空间等
    if (reverse_dimensions()) {
        conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
    } else {
        conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
    }

    //卷积窗口在输入“图像”上按步长滑动，形成了多个子图;然后将所有子图拉成一列，列的长度就是col_offset_。
    //col_offset_与im2col_cpu()函数中channels_col的计算是相似的，但是值并不相等，
    //原因在于：channels_col是将卷积层输入的通道数conv_in_channels_用于相乘，
    //但kernel_dim_只用到了一部分channel,也就是分组卷积，conv_in_channels_/group_ 。
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    //卷积层的输出特征图也要分组，当然group_默认为1。写成(conv_out_channels_ / group_) * conv_out_spatial_dim_更直观
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_; // C*H*W/group_

    //Setup input dimensions(conv_input_shape_)
    vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
    conv_input_shape_.Reshape(bottom_dim_blob_shape);

    int * conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
    for (int i = 0; i < num_spatial_axes_ +1; ++i) {
        if(reverse_dimensions()) {
            conv_input_shape_data[i] = top[0] ->shape(channel_axis_ +i);
        } else {
            conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
        }
    }

    // The im2col result buffer will only hold one image at a time to avoid
    // overly large memory usage. In the special case of 1x1 convolution
    // it goes lazily unused to save memory.
    col_buffer_shape_.clear();
    col_buffer_shape_.push_back(kernel_dim_ * group_);
    for (int i = 0; i < num_spatial_axes_; ++i) {
        if (reverse_dimensions()) {
            col_buffer_shape_.push_back(input_shape(i + 1));
        } else {
            col_buffer_shape_.push_back(output_shape_[i]);
        }
    }
    col_buffer_.Reshape(col_buffer_shape_);
    bottom_dim_ = bottom[0]->count(channel_axis_);
    top_dim_ = top[0]->count(channel_axis_);
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
    // Set up the all ones "bias multiplier" for adding biases by BLAS
    out_spatial_dim_ = top[0]->count(first_spatial_axis);
    if (bias_term_) {
        vector<int> bias_multiplier_shape(1, out_spatial_dim_);
        bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());
    }
} //close Reshape


/**
 *
 * 前向传播 cpu 矩阵相乘, 主要处理weight 与输入矩阵的乘法*
 * forward_cpu_gemm完成的主要操作为 将input 进行im2col变换，
 * 得到col_buffer_, 然后与weights相乘，得到的结果存放在output中
 *
 * */

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
        const Dtype* weights, Dtype*output, bool skip_im2col){
    const Dtype* col_buff = input;

    //如果非1x1和省略im2col的话
    if(!is_1x1_) {
        if (!skip_im2col) {
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
        }
        col_buff = col_buffer_.cpu_data();
    }

    for (int g=0; g<group_; ++g){
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1, weights +weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
    //前面两个CblasNoTrans 表示A和B都不需要转置
    //conv_out_channels_ / group_ 表示A和C矩阵的行数row,  也就是M， group_ 默认为1
    //conv_out_spatial_dim_ 表示B和C的列数 也就是N
    //kernel_dim_ 表示A的列及B的行
    //Dtype 1. 为alpha
    //Dtype 0. 为beta 表示公式为 C = A*B
    //weight + weight_offset_ * g = 矩阵A, weight是指针 [MxK]
    //col_buff + col_offset_ * g = 矩阵B col_buff是指针 [KxN]
    //output + output_offset_ * g = 矩阵C output 是指针 [MxN]

    //在这个函数中，weight_offset_ = channels*卷积核个数*卷积核width*卷积核height，
    // col_offset_的值就是我在im2col_cpu中计算的data_col的大小，
    // output_offset_ = 卷积核个数*卷积后的图片宽度*卷积后的图片长度。
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                   const Dtype* bias) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
                          (Dtype)1., output);
}



template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype *input,
        const Dtype *weights, Dtype *output) {
    Dtype* col_buff = col_buffer_.mutable_cpu_data(); //col_buffer_ 为 im2col函数产出的列向量

    if (is_1x1_) {
        col_buff = input; //如果是1x1卷积的话， 做了im2col之跟Input都是一样的， 所有省略im2col直接赋值input
    }
    for (int g = 0; g<group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                conv_out_spatital_dim_, conv_out_channels_ / group_,
                (Dtype)1., weigths + weight_offset_ *g, output _ output_offset_ * g),
                (Dtype)0., col_buff + col_offset_ * g);
    }

    if (!is_1x1_){
        conv_col2im_cpu(col_buff, input); //如果不是1x1卷积， 就直接im2col， 将列向量与input相乘
    }

}

/**
 * 反向传播计算关于权重的导数用于更新权重
 * **/
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype * input,
        const Dtype* output, Dtype* weights){
    const Dtype * col_buff = input;
    if (!is_1x1_) {
        conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
        col_buff = col_buffer_.cpu_data();
    }
    for (int g = 0; g <group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                kernel_dim_, conv_out_spatial_dim_,
                (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
                (Dtype)1., weights + weight_offset_ * g);
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                    const Dtype* input) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
        const Dtype* weights, Dtype*output, bool skip_im2col) {
    const Dtype* col_buff = input;
    if(!is_1x1_) {
        if(!skip_im2col) {
            conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
        }
        col_buff = col_buffer_.gpu_data();
    }
    for (int g = 0; g<group_; ++g){
        caffe_gpu_gemm<Dtype>(CblasNoTran, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_, (Dtype)1.,
        weights + weight_offset_ * g, col_buff + col_offset_ * g,
                (Dtype)0., output + output_offset_ * g);
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
                                                   const Dtype* bias) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
                                                    const Dtype* weights, Dtype* input) {
    Dtype* col_buff = col_buffer_.mutable_gpu_data();
    if (is_1x1_) {
        col_buff = input;
    }
    for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                              conv_out_spatial_dim_, conv_out_channels_ / group_,
                              (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
                              (Dtype)0., col_buff + col_offset_ * g);
    }
    if (!is_1x1_) {
        conv_col2im_gpu(col_buff, input);
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
                                                  const Dtype* output, Dtype* weights) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
        conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
        col_buff = col_buffer_.gpu_data();
    }
    for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                              kernel_dim_, conv_out_spatial_dim_,
                              (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
                              (Dtype)1., weights + weight_offset_ * g);
    }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
                                                    const Dtype* input) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif // CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);


}//namespace caffe close
