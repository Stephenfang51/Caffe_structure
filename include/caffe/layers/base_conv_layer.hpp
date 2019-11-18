#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "../blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"
#include "../util/im2col.hpp"

namespace caffe {
/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */


template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
public:
    explicit BaseConvolutionLayer(const LayerParameter & param):Layer<Dtype>(param) {}
    //构造函数， 同时为基类Layer提供数据初始化

    virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top);
    //重写LayerSetUp方法

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //重写Reshape方法

    virtual inline int MinBottomBlobs() const {return 1;}
    virtual inline int MinTopBlobs() const {return 1;}
    virtual inline bool EqualNumBottomTopBlobs() const {return true;}

protected:

    //gemm = general matrix matrix multiplication
    //最后一个参数 skip_im2col 可以设定略过im2col 如果只call weight_cpu_gemm with the same input
    void forward_cpu_gemm(const Dtype* input, const Dtype*weights, Dtype* output, bool skip_im2col = false);
    void forward_cpu_bias(Dtype* output, const Dtype* bias);
    void backward_cpu_gemm(const Dtype* input, const Dtype * weights, Dtype* output);
    void weight_cpu_gemm(const Dtype*input, const Dtype* output, Dtype* weights);
    void backward_cpu_bias(const Dtype* bias, const Dtype*input);

#ifndef CPU_ONLY
    void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights, Dtype* output, bool skip_im2col = false)
    void forward_gpu_bias(Dtype* output, const Dtype* bias);
    void backward_gpu_gemm(const Dtype* input, const Dtype * weights, Dtype* col_output);
    void weight_gpu_gemm(const Dtype* col_input, const Dtype*output, Dtype * weights);
    void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

    ///@brief 输入i 返回维度
    inline int input_shape(int i){
        return (*bottom_shape_)[channel_axis_ + i];
    }

    //如果是deconv 则返回True,
    virtual bool reverse_dimensions() = 0;
    //计算出height_out_ and width_out_ from other parameters
    virtual void compute_output_shape() = 0;


    /// @brief The spatial dimensions of a filter kernel.
    Blob<int> kernel_shape_;
    /// @brief The spatial dimensions of the stride.
    Blob<int> stride_;
    /// @brief The spatial dimensions of the padding.
    Blob<int> pad_;
    /// @brief The spatial dimensions of the dilation.
    Blob<int> dilation_;
    /// @brief The spatial dimensions of the convolution input.
    Blob<int> conv_input_shape_;
    /// @brief The spatial dimensions of the col_buffer.
    vector<int> col_buffer_shape_;
    /// @brief The spatial dimensions of the output.
    vector<int> output_shape_;
    const vector<int>* bottom_shape_;

    int num_spatial_axes_; //空间轴的数量 H and W就是2个
    int bottom_dim_; // 输入度维度 = 输入图像通道数*输入图像的h*输入图像w
    int top_dim_; // 输出维度 = 输出通道数*输出h*输出w
    int channel_axis_; //通道的轴
    int num_; //batch_size
    int channels_; // 输入图像的通道数
    int group_;//卷积组大小
    int out_spatial_dim_;  // 输出空间维度 = 卷积之后的图像长*卷积之后图像的宽
    int weight_offset_; //channels*卷积核个数*卷积核w*卷积核h
    int num_output_;// 卷积后的图像的通道数
    bool bias_term_;// 是否启用偏置
    bool is_1x1_;// 是不是1x1卷积
    bool force_nd_im2col_;// 强制使用n维通用卷积

/**im2col_cpu 传入参数依序是**/

//data_im 数据指针
//channels 输入通道数
//height / width 输入图像尺寸
//kernel_h / kernel_w 卷积核尺寸
//pad_h / pad_w padding值
//stride_h / stride_w 滑动步长
//dilation_h / dilation_w 空洞卷积 一般卷积时 为1
//data_col 输出矩阵数

private:
    inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
        if(!force_nd_im2col_ && num_spatial_axes_ == 2 ) {
            //调用im2col_cpu function （from im2col.hpp)
            im2col_cpu(data, conv_in_channels_, conv_input_shape_.cpu_data()[1],
            conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_().cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
        }
        else{ //2维以上的卷积
            im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                    col_buffer_shape_.data(), kernel_shape_.cpu_data(), pad_.cpu_data(),
                    stride_.cpu_data(), dilation_.cpu_data(), col_buff);
        }
    }
    //反推
    inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data){

        //判断维度为2或者是以上， 调用不同的function
        if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
            col2im_cpu(col_buff, conv_in_channels_, conv_input_shape_.cpu_data()[1],
            conv_input_shape_.cpu_data()[2],
            kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
        } else {
            col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
                          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
        }
    }
#ifndef CPU_ONLY // 实现GPU的部分
inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {

    //判断维度为2或者是以上， 调用不同的function
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        im2col_gpu(data, conv_in_channels_,
                   conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                   kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                   dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
        im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
}
inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {

    //判断维度为2或者是以上， 调用不同的function
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        col2im_gpu(col_buff, conv_in_channels_,
                   conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                   kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                   dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
        col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
                      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                      kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
                      dilation_.gpu_data(), data);
    }
}
#endif
    /**
     * 主要定义
     *
     * */
        int num_kernels_im2col_; // conv_in_channels_ * conv_out_spatial_dim_
        int num_kernels_col2im_; //num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_
        int conv_out_channels_; // 卷积的输出通道数 ,在参数配置文件中设置
        int conv_in_channels_; // 卷积的输入通道数 （即输入图像的通道数）
        int conv_out_spatial_dim_; // 卷积的输出的空间维度 = 卷积后图像h*卷积后图像w
        int kernel_dim_; // 卷积核的维度 = 输入图像的维度*卷积核的h*卷积核的w
        // 在使用gropu参数的时候使用的offset
        int col_offset_;
        int output_offset_; //卷积核个数*卷积后图像h*卷积后图像w CHW

        Blob<Dtype> col_buffer_;// im2col的时候使用的存储空间
        Blob<Dtype> bias_multiplier_;



}; //class BaseConvolutionLayer close


}//namespace caffe close