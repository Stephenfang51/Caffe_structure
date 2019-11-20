#include <algorithm>
#include <vector>

#include "../../../include/caffe/layers/relu_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) {
        const Dtype *bottom_data = bottom[0]->cpu_data;
        Dtype *top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();


        //ReLU激活函数极为f(x)=max(0,x)。
        //negative_slope 如果是ReLU的话则为0， LeakyReLU则为

        for (int i = 0; i < count; ++i) { //遍历每一个元素，
            top_data[i] = std::max(bottom_data[i], Dtype(0))
                          //两者取最大值， 小于0的话就会取0
                          + negative_slope * std::min(bottom_data[i], Dtype(0));
            //ReLU时， negative_slope取0， 该项消除
        };

    }
template <typename Dtype>
    void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                        const vector<bool> & propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {
        if (propagate_down[0]) {
            const Dtype* bottom_data = bottom[0]->cpu_data;
            const Dtype * top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
            for (int i = 0; i < count; ++i){

                //ReLU 导数 当x< 0 则 导数为0， 当x>0 则导数为1(完整传播）， 导数由bottom_data[i]>0 返回0 or 1判断
                //bottom_data[i] 可以视为输入x
                //top_diff 为上一层 乘上 relu的导数 0 or 1，
                
                bottom_diff[i] = top_diff[i] * ((bottom_data[i]>0)
                        + negative_slope * (bottom_data[i]<= 0));
            }
        }
    }


#ifdef CPU_ONLY
STUB_GPU(ReLUayer);
#endif
INSTANTIATE_CLASS(ReLULayer);

}//namespace caffe closed

