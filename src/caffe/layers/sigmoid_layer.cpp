#include <cmath>
#include <vector>

#include "../../../include/caffe/layers/sigmoid_layer.hpp"

namespace caffe {
template <typename Dtype>

//先行定义sigmoid公式
inline Dtype sigmoid(Dtype x) {
    return 0.5 * tanh(0.5 * x) + 0.5;
}

template  <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    //取得l和l-1层的导数， l层 是可修改的
    const Dtype * bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0] ->mutable_cpu_data();


    const int count = bottom[0] ->count();
    for (int i = 0; i<count; ++i){
        top_data[i] = sigmoid(bottom_data[i]);
    }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top, const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype>*> & bottom) {
    if (propagate_down[0]) {
        //取得 l 层的数据和导数，  l-1层的导数
        const Dtype * top_data = top[0]->cpu_data();
        const Dtype * top_diff = top[0]->cpu_diff();
        Dtype * bottom_diff =  bottom[0]->count();
        const int count = bottom[0] ->count();

        for (int i = 0; i < count; ++i) {
            const Dtype sigmoid_x = top_data[i];
            bottom_diff[i]  = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
            //sigmoid 导数 : S'(x) = S(x) * (1-S(x))
            //那么由l (top)传到 l-1(bottom) 就是top 乘上偏微分之后的值


    /**
     * python实现逻辑
     *
     * def sigmoid(z, derivative=False):
           sigmoid = 1.0/(1.0+np.exp(-z))
           if (derivative==True):
             return sigmoid * (1-sigmoid)
           return sigmoid
     */
        }

    }
}
#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);

} //namesace caffe closed

