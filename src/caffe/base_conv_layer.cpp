#include <algorithm>
#include <vector>
//#include "caffe/filler.hpp"
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
        const vector<Blob<Dtype>*>& top){
    //计算出kernel size, padding, stride and inputs
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    force_nd_im2col_ = conv_param.force_nd_im2col(); //force_nd_im2col 是 类方法
    channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis())//blob 调用CanonicalAxisIndex(), 负变为正
    const int first_spatial_axis = channel_axis_ +1; //定义第一个空间(H, W)轴， 也就是height的index？
    const int num_axes = bottom[0] -> num_axes();//blob 调用num_axes返回维度
    num_spatial_axes_ = num_axes - first_spatial_axis; //计算出空间轴的个数 也就是HW 2D应该为2个
    CHECK_GE(num_spatial_axes_, 0); //确定上一行代码计算出的大于等于0
    vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
    //vector 存放1个元素， 初始值为std::max(num_spatial_axes_, 1)
    /*****************设置kernel 维度 (kernel_shape_)************************/
    kernel_shape_.Reshape(spatial_dim_blob_shape);//设定好卷积核的维度, 记住kernel_shape_是blob类型
    int * kernel_shape_data = kernel_shape_.mutable_cpu_data();//调用可读写存在cpu上数据
    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()){
        //确认有卷积核的H， W

        CHECK_EQ(num_spatial_axes_, 2)
            <<"kernel_h & kernel_w can only be used for 2D convolution.";
        CHECK_EQ(0, conv_param.kernel_size_size())
            <<"Either kernel_size or kernel_h/w should be specified; not both.";
        kernel_shape_data[0] = conv_param.kernel_h(); //设置卷积核的H， W
        kernel_shape_data[1] = conv_param.kernel_h();
    } else {
        //否则的话， 从proto中取得kernel的尺寸
        //下面看不懂

        const int num_kernel_dims = conv_param.kernel_size_size();
        CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
            <<"kernel_sie must be specified once, or once per spatial dimension "
            << " (kernel_size specified " << num_kernel_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";

        for (int i = 0; i < num_spatial_axes_; ++i){
            //根据num_spatial_axes 的总数 赋值给kernel_shape_data[i]
            kernel_shape_data[i] =
                    conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
        //上面看不懂
        }
    }

    for (int i = 0; i < num_spatial_axes_; ++i){
        //确认kernel的元素， 每一个 val1 大于  val2, 不能为0
        CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
    }

    /**************设置好 步长维度 (stride_)*****************/
    stride_.Reshape(spatial_dim_blob_shape); //注意stride_为blob类型，依照spatial_dim_blob_shape设置

    int * stride_data = stride_.mutable_cpu_data();//stride_data 指针 指向blob调用的可读写cpu数据

    if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
            <<"stride_h & stride_w can only be used for 2D covolution.";
        CHECK_EQ(0, conv_param.stride_size())
            <<"Either stride or stride_h/w should be specified; not both";
        stride_data[0] = conv_param.stride_h();
        stride_data[1] = conv_param.stride_w();
    }else {
        const int num_stride_dims = conv_param.stride_size();
        CHECK(num_stride_dims == 0 || num_stride_dims == 1||
            num_stride_dims == num_spatial_axes_)
            << "stride must be specified once, or once per spatial dimension "
            << "(stride specified " << num_stride_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";

        const int kDefaultStride = 1; // 默认步长
        for (int i = 0; i < num_spatial_axes_; ++i){ //依照空间轴数赋值
            stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                    conv_param.stride((num_stride_dims == 1) ? 0 : i);
            CHECK_GT(stride_data[i], 0) << "Stride demensions must be nonzero.";
        }
    }

    /*****************设置pad 维度************************/
    pad_.Reshape(spatial_dim_blob_shape);
    int* pad_data = pad_.mutable_cpu_data();
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
    int* dilation_data = dilation_.mutable_cpu_data();
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
    for (int i =0 ; i < num_spatial_axes_; ++i){
        is_1x1_ &= //&= 按位“与”运算
                kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
        if(!is_1x1_){break;}
    }
    /**计算output channel and groups**/
    channels_ = bottom[0] ->shape(channel_axis_); //blob调用shape方法， 带入index取得相应的值
    num_output_ = this->layer_param_.convolution_param().num_output();
    //class layer调用protected 成员layer_param_ 并调用
    //proto中LayerParameter的方法 convolution_param()取得卷积层的参数并取得卷积核的个数

    CHECK_GT(num_output_, 0); //卷积核个数大于0

    //group的部分
    group_ = this->layer_param_.convolution_param().group();
    CHECK_EQ(channels_ % group_, 0); //意味着必须channels_除group_整除
    CHECK_EQ(num_output_% group_,  0)  //意味着必须num_output_除group_整除
        << "Number of output should be multiples of group.";

    //判断是否为反卷积， 并作出相应的赋值
    if (reverse_dimensions()) {
        conv_out_channels_ = channels_;
        conv_in_channels_ = num_output_;
    } else {
        conv_out_channels_ = num_output_;
        conv_in_channels_ = channels_;
    }

    //处理参数 ： weight 及 bias 偏值
    // blobs_[0] 对应 卷积核上的weights
    // blobs_[1] 对应 偏值（可选）

    vector<int> weight_shape(2); //创建一个vector容器， 大小为2
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for ( int i = 0; i < num_spatial_axes_; ++i){
        weight_shape.push_back(kernel_shape_data[i]);
        //kernel_shape_data是int指针， 将kernel shape存放到weight_shape后面
    }

    bias_term_ = this->layer_param_.convolution_param().bias_term();
    vector<int> bias_shape(bias_term_, num_output_);
    //创建一个bias_shape容器， 大小为bias_term_ 非0即1


    //找不到blobs come from ??
    if(this->blobs_.size()>0) {
        CHECK_EQ(1 + bias_term_, this->blobs_.size())
            << "Incorrect number of weight blobs.";
        if (weight_shape != this-> blobs_[0]->shape()){
        //
            Blob<Dtype> wegiht_shaped_blob(weight_shape);
            LOG(FATAL) << "Incorrect weight shape: expeectd shape "
                << weight_shaped_blob.shape_string() << "； instead， shape was "
                << this->blobs_[0] ->shape_string();
        }
        if (bias_term_ && bias_shape)
    }












}//LayerSetUp close

}//namespace caffe close
