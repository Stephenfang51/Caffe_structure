#include <vector>
#include "../../../include/caffe/layers/conv_layer.hpp"

namespace caffe {
template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
    const int * kernel_shape_data = this->kernel_shape_.cpu_data();
    const int * stride_data = this ->stride_.cpu_data();
    const int * pad_data = this->pad_.cpu_data();
    const int * dilation_data = this->dilation_.cpu_data();
    this->output_shape_.clear();


    // num_spatial_axes = 2 H/W, 依序将spatial axes 的H 和 W依据公式推算出 output feature map
    for (int i = 0; i <this ->num_spatial_axes_; ++i){

        //i + 1 to skip channel axis， 依次处理Height and Width
        const int input_dim = this->input_shape(i+1);


        const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] -1) +1;
        //比照公式计算出output feature map
        const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                / stride_data[i] + 1;

        this ->output_shape_.push_back(output_dim);

    }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
      const vector<Blob<Dtype> *> &top) {

    const Dtype * weight = this->blobs_[0]->cpu_data();
    //blobs_[0] 为weight
    //blobs_[1] 为 bias

    for (int i = 0; i < bottom.size(); ++i){

        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype * top_data = top[i]->mutable_cpu_data();

        for (int n = 0; n<this->num_; ++n) //num_ = batch_size
            this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                    top_data + n * this->top_dim_);
    //forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype*output, bool skip_im2col=false)
    //bottom_dim_ = 输入维度（channels * H * W)

    }
}


template <typename Dtype>
//实现反向传播，根据上一层传下来的导数计算相应的bottom data ， weight， bias 的导数
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom){
    //取得weight 和 导数
    //反向传播梯度误差
    const Dtype*weight = this->blobs_[0].cpu_data;
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

    //因为是反向传播 从top的地方开始
    for (int i = 0; i< top.size(); ++i){
        const Dtype* top_diff = top[i]->cpu_diff(); //取得top的导数

        const Dtype* bottom_data = bottom[i]->cpu_data(); //取得下一层的
        Dtype * bottom_diff = bottom[i] ->mutable_cpu_diff();//取得下一层的梯度， 需要更新的


        //如果有bias项，计算Bias导数
        if (this->bias_term_ && this->param_propagate_down_[1]){
            Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
            for (int n = 0; n < this->num_ ; ++n){
                this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
            }
        }
        //计算weight
        if ( this->param_propagate_down_[0] || propagate_down[i]){
         for (int n = 0; n<this->num_; ++n){
             // 计算weights权重的梯度

             if (this->param_propagate_down_[0]){
                 this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                       top_diff + n * this->top_dim_, weight_diff);
             }
             // gradient w.r.t. bottom data, if necessary.
             //计算botttom数据的梯度，下后传递
             if (propagate_down[i]) {
                 this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                         bottom_diff + n * this->bottom_dim_);
             }
            }
        }
    }
#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

} // namespace caffe closed


