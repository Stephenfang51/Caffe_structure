#ifndef CAFFE_DETAIL_LAYER_HPP
#define CAFFE_DETAIL_LAYER_HPP

#include <algorithm>
#include <string>
#include <vector>

#include "../caffe/blob.hpp"
#include "common.hpp"
#include "layer_factory.hpp"
#include "proto/caffe.pb.h"
#include "util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */

namespace boost {class mutex; }

namespace caffe {
    /**
     *
     * @brief  An interface for the units of computation which can be composed into a
     *        Net.
     *
     * Layer 必须执行Forward function以及backward function   以及计算出loss
     */

template <typename Dtype>
class Layer {
public:
    p
    //尝试从protobuf文件读取参数
    {
        //设定阶段及复制blobs
        phase_ = params.phase(); //phase_ 是枚举类型
        if (layer_param_.blobs_size()>0) {  //如果有参数值
            blobs_.resize(layer_param_.blobs_size()); //则blobs_ resize成和参数一样

            //从proto存储的变为blobs_， #blobs_ 为vector存了shared_ptr指向Blob
            for (int i = 0; i < layer_param_.blobs_size(); ++i){
                blobs_[i].reset(new Blob<Dtype>());
                blobs_[i]->FromProto(layer_param_.blobs(i));
                //调用Blob类中的FromProto, 反序列化
            }
        }
    }
    virtual ~Layer(){}
    /**
  * @brief Implements common layer setup functionality.
  *
  * @param bottom the preshaped input blobs
  * @param top
  *     the allocated but unshaped output blobs, to be shaped by Reshape
  *
  * Checks that the number of bottom and top blobs is correct.
  * Calls LayerSetUp to do special layer setup for individual layer types,
  * followed by Reshape to set up sizes of top blobs and internal buffers.
  * Sets up the loss weight multiplier blobs for any non-zero loss weights.
  * This method may not be overridden.
  */
    void SetUp(const vector<Blob<Dtype>*> & bottom,
            const vector<Blob<Dtype>*> & top) {
        CheckBlobCounts(bottom, top);
        LayerSetUp(bottom, top);
        Reshape(bottom, top);
        SetLossWeights(top);
    }

    //在各种派生类中自行定义，用来执行解析layer parameters以及判断bottom blob shape是否正确等初始化操作
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top)
    /**
* @brief Adjust the shapes of top blobs and internal buffers to accommodate
*        the shapes of the bottom blobs.
*
* @param bottom the input blobs, with the requested input shapes
* @param top the top blobs, which should be reshaped as needed
*
* This method should reshape top blobs as needed according to the shapes
* of the bottom (input) blobs, as well as reshaping any internal buffers
* and making any other necessary adjustments so that the layer can
* accommodate the bottom blobs.
*/
    //必须要重新设置top blob, 此hpp中未定义
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top) = 0;


    /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
    inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);




    /**
 * @brief Given the top blob error gradients, compute the bottom blob error
 *        gradients.
 *
 * @param top
 *     the output blobs, whose diff fields store the gradient of the error
 *     with respect to themselves
 * @param propagate_down
 *     a vector with equal length to bottom, with each index indicating
 *     whether to propagate the error gradients down to the bottom blob at
 *     the corresponding index
 * @param bottom
 *     the input blobs, whose diff fields will store the gradient of the error
 *     with respect to themselves after Backward is run
 *
 * The Backward wrapper calls the relevant device wrapper function
 * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
 * top blob diffs.
 *
 * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
 */
    inline void Backward(const vector<Blob<Dtype>*>& top,
                         const vector<bool>& propagate_down,
                         const vector<Blob<Dtype>*>& bottom);



    //返回可学习参数
    vector<shared_ptr<Blob<Dtype> > >& blobs() {
        return blobs_;
    }

    //返回该layer的参数
    const LayerParameter& layer_param() const { return layer_param_; }

    //序列化
    virtual void ToProto(LayerParameter* param, bool write_diff = false);

    /**
     * @brief Returns the scalar loss associated with a top blob at a given index.
     */
     //给指定的index返回输出blob的scalar loss_
    inline Dtype loss(const int top_index) const {
        return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
    }

    /**
     * @brief Sets the loss associated with a top blob at a given index.
     */
     //在指定的index上设定输出blob的loss值
    inline void set_loss(const int top_index, const Dtype value) {
        if (loss_.size() <= top_index) {
            loss_.resize(top_index + 1, Dtype(0));
        }
        loss_[top_index] = value;
    }

    //返回layer的type
    virtual inline const char* type() const { return ""; }

    virtual inline int ExactNumBottomBlobs() const {return -1;}
    //返回bottom blobs正确数量， -1代表不需要正确数量
    virtual inline int MinBottomBlobs() const { return -1; }
    //返回bottom blobs最小需要数量， -1代表不需要正确数量
    virtual inline int MaxBottomBlobs() const { return -1; }
    //返回bottom blobs最大需要的数量， -1代表不需要正确数量
    virtual inline int ExactNumTopBlobs() const { return -1; }
    //返回top blobs正确数量， -1代表不需要正确数量
    virtual inline int MinTopBlobs() const { return -1; }
    //返回top blobs最小需要数量， -1代表不需要正确数量
    virtual inline int MaxTopBlobs() const { return -1; }
    //返回top blobs最大需要数量， -1代表不需要正确数量

    virtual inline bool EqualNumBottomTopBlobs() const { return false; }

    virtual inline bool AutoTopBlobs() const { return false; }






    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
    }//是否强制梯度返回

    /**
 * @brief Specifies whether the layer should compute gradients w.r.t. a
 *        parameter at a particular index given by param_id.
 *
 * You can safely ignore false values and always compute gradients
 * for all parameters, but possibly with wasteful computation.
 */
    inline bool param_propagate_down(const int param_id) {
        return (param_propagate_down_.size() > param_id) ?
               param_propagate_down_[param_id] : false;
        //利用？ 条件式， 如果传入的参数比param_propagate小
        //则该index 设置为 false 表示不用回传
    }


    /**
     * @brief Sets whether the layer should compute gradients w.r.t. a
     *        parameter at a particular index given by param_id.
     */
    inline void set_param_propagate_down(const int param_id, const bool value) {
        if (param_propagate_down_.size() <= param_id) {
            param_propagate_down_.resize(param_id + 1, true);
        }
        param_propagate_down_[param_id] = value;
    }
protected:
    LayerParameter layer_param_; //保存网络层参数, LayerParameter 是caffe.pb.h定义的类
    Phase phase_;//TRAIN = 0 or TEST = 1 Phase 定义在caffe.ph.h中是枚举
    vector<shared_ptr<Blob<Dtype> > > blobs_; //vector容器保存了learnable parameter(weight and bias)
    vector<bool> param_propagate_down_;//vector index指出哪些param要计算梯度

    vector<Dtype> loss_; //指出是否每一个top blob都有 非0 的weight 在损失函数

    //使用CPU， 计算网络层的output, 可以看见分成bottom and top
    virtual void Forward_cpu(
            const vector<blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) = 0;

    virtual void Forward_gpu(
            const vector<blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        //LOG(WARNING)<< "Using CPU code as backup.";
        return Forward_cpu(bottom, top);
    }

    //使用CPU, 计算每一个参数的梯度并且如果propagate = true, bottom blobs也要
            const vector<Blob<Dtype>*>&bottom) = 0;
    //使用GPU
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propaagate_down,
            const vector<Blob<Dtype>*> & bottom){
        //LOG(WARNING） << "Using CPU code as backup.";
        Backward_cpu(top, propagate_down, bottom);
    }

    //该函数避免输入错误的blobs尺寸
    virtual void CheckBlobCounts(const vector<Blob<Dtype>*> & bottom,
            const vector <Blob<Dtype>*>& top){
        if (ExactNumBottomBlobs() >= 0){
            CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                << type() << " Layer takes " << ExactNumBottomBlobs()
                <<" bottom blob(s) as input.";
        }
        if (MinBottomBlobs()>= 0){
            CHECK_LE(MinBottomBlobs(), bottom.size())
                << type() << " Layer takes at least " << MinBottomBlobs()
                << " bottom blob(s) as input.";
        }
        if (MaxBottomBlobs() >= 0) {
            CHECK_GE(MaxBottomBlobs(), bottom.size())
                << type() << " Layer takes at most " << MaxBottomBlobs()
                << " bottom blob(s) as input.";
        }
        if (ExactNumTopBlobs() >= 0) {
            CHECK_EQ(ExactNumTopBlobs(), top.size())
                << type() << " Layer produces " << ExactNumTopBlobs()
                << " top blob(s) as output.";
        }
        if (MinTopBlobs() >= 0) {
            CHECK_LE(MinTopBlobs(), top.size())
                << type() << " Layer produces at least " << MinTopBlobs()
                << " top blob(s) as output.";
        }
        if (MaxTopBlobs() >= 0) {
            CHECK_GE(MaxTopBlobs(), top.size())
                << type() << " Layer produces at most " << MaxTopBlobs()
                << " top blob(s) as output.";
        }
        if (EqualNumBottomTopBlobs()) {
            CHECK_EQ(bottom.size(), top.size())
                << type() << " Layer produces one top blob as output for each "
                << "bottom blob input.";
        }
    }

    /**
     * Any layer can be used as a loss by adding a field loss_weight
     * 初始化top bottom的weights，并且存储非零的loss weights 在diff blob里面
     */
    inline void SetLossWeights(const vector<Blob<Dtype>*>)& top){
        const int num_loss_weights = layer_param_.loss_weight_size();
        if (num_loss_weights){
            CHECK_EQ(top.szie(), num_loss_weights) << "loss_weight must be "
              "unspecified or specified once per top blob.";
            for (int top_id = 0; top_id < top.size(); ++top_id) {
                const Dtype loss_weight = layer_param_.loss_weight(top_id);
                if (loss_weight == Dtype(0)) {continue;}
                this->set_loss(top_id, loss_weight);
                const int count = top[top_id] ->count();
                Dtype* loss_multiplier = top[top_id] -> mutable_cpu_diff();
                caffe_set(count, loss_weight, loss_multiplier);
                //caffe_set 在math_functions

               }
            }
        }

    private:
    DISABLE_COPY_AND_ASSIGN(Layer);
}; // class Layer



//Forward and backward wrappers.
//Forward wrappers
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*& bottom,
        const vector<Blob<Dtype>*>& top){
    Dtype loss = 0;//初始化为0
    Reshape(bottom, top);
    switch(Caffe::mode()) { // 依照当前的mode进行计算
    case Caffe::CPU: //使用CPU进行计算时
        Forward_cpu(bottom, top);//调用函数
        for (int top_id = 0; top_id < top.size(); ++top_id){
            if(!this->loss(top_id)) {continue;}
            const int count top[top_id] ->count();
            const Dtype*data = top[top_id] ->cpu_data();
            const Dtype*loss_weights = top[top_id] ->cpu_diff();
            loss += caffe_cpu_dot(count, data, loss_weights);
        }


    break;
    case Caffe::GPU: //使用GPU进行计算时
        Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id){
        if (!this->loss(top_id) {continue;}
        const int count = top[top_id] ->count();
        const Dtype* data = top[top_id]->gpu_data();
        const Dtype* loss_weights = top[top_id]->gpu_diff();
        Dtype blob_loss = 0;
        caffe_gpu_dot(count, data, loss_weights, &blob_loss);
        loss += blob_loss;
    }
#endif
    break;
        default:
            LOG(FATAL) << "Unknown caffe mode.";
}
    return loss;

}

//Backward
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
        const vector<bool> & propagate_down,
        const vector<Blob<Dtype>*>)& bottom){
        switch (Caffe::mode()){ //视mode 调用cpu or gpu
        case Caffe::CPU:
            Backward_cpu(top, propagate_down, bottom);
            break;
        case Caffe::GPU:
            Backward_gpu(top, propagate_down, bottom);
            break;
        default:
            LOG(FATAL) << "Unkown caffe mode.";
        }
}

template <typename Dtype> //序列化 : 将Layer参数写入到protobuff
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff){
            param->Clear(); //清除
            param->CopyFrom(layer_param_);//从layer_param_复制
            param->clear_blobs();
            for (int i=0; i < blobs_.size(); ++i){
                blobs_[i]->ToProto(param->add_blobs(), write_diff);
            }
        }





}// namepsace caffe

#endif //CAFFE_DETAIL_LAYER_HPP