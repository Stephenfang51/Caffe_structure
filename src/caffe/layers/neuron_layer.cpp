#include <vector>
#include "../../../include/caffe/layers/neuron_layer.hpp"

namespace caffe {
template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top){
    top[0] ->ReshapeLike(*bottom[0]);
}

}//namespace caffe closed

