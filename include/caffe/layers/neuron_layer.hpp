//
// Created by StephenFang on 2019/11/20.
//

#ifndef CAFFE_DETAIL_NEURON_LAYER_HPP
#define CAFFE_DETAIL_NEURON_LAYER_HPP


#include "../blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {
    /**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
public:
    explicit NeuronLayer(const LayerParameter & param) : Layer<Dtype>(param) {}
    virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
            const vector<Blob<Dtype>* >&top);


    virtual inline int ExactNumBottomBlobs() const {return 1;}
    virtual inline int ExactNUmTopBlobs() const {return 1;}

};

} //namespace caffe closed





#endif //CAFFE_DETAIL_NEURON_LAYER_HPP
