/**
 * 一个layer factor 允许注册一层， 被注册的层可以被呼叫 藉由passing 一层LayerParamemter
 * protobuffer to the CreateLayer function:
 *
 *          LayerResigtry<Dtype>::CreateLayer(param);
 * 有两个方法可以注册一个layer, 假设我们有以下的层
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 *   那么它的类型就是它类名但是去掉最后面的Layer
 *   ("MyAwesomeLayer" -> "MyAwesome").
 *
 *   如果是用上面这种方式就利用#define
 *   REGISTER_LAYER_CLASS(MyAwesome);
 *
 *   或者如果使用另种方式创造一个网络层的话
 *   template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 *    举个例子， 当你的layer has multiple backends, 例如像是GetConvolutionLayer
 *    那么你可以下面这样
 *    REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 *    注意: 记得每一种layer type 只能被registered once
 *
 */

///部分参考
/// https://blog.csdn.net/fengbingchun/article/details/54310956
/// https://blog.csdn.net/xizero00/article/details/50923722

#ifndef CAFFE_DETAIL_LAYER_FACTORY_HPP
#define CAFFE_DETAIL_LAYER_FACTORY_HPP


#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "layer.hpp"
#include "proto/caffe.pb.h"

namespace caffe {
template<typename Dtype>
class Layer;

template<typename Dtype>
class LayerRegistry {
public:
    ///智能指针指向Layer<Dtype>, 创建名为Creator的指针
    typedef shared_ptr<Layer<Dtype>> (*Creator) (const LayerParameter&);
    typedef std::map<string, Creator> CreatorRegistry;
    ///map创建容器 也就是一个注册表， 并且重名为CreatorResgistry

    ///静态成员函数, 定义一个Registry注册方法， 用来获取全局单例
    static CreatorRegistry & Registry() {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        ///注册过程就是map操作， 指针的方式创建动态内存
        return *g_registry_;
    }
    // Adds a creator.
    ///AddCreator函数用来向Registry列表中添加一组<type, creator>
    ///参数1 指定key 参数2指定value
    ///在map中加入一组映射
    ///Creator 是指针类
    static void AddCreator(const string& type, Creator creator) {
        CreatorRegistry& registry = Registry(); ///获取注册表指针
        CHECK_EQ(registry.count(type), 0)///确认要创建的先前不存在
            << "Layer type " << type << " already registered.";
        registry[type] = creator; ///向Registry列表中添加一组<layername, creatorhandlr>
    }

    ///这个创建层在net.cpp中会用到，在初始化整个网络的时候会根据参数文件中的层的类型去创建该层的实例
    static shared_ptr<Layer<Dtype>> CreateLayer (const LayerParameter & param) {
        if (Caffe::root_solver()) {
            LOG(INFO) << "Creating layer " << param.name();
        }

        const string & type = param.type();///从参数中获得类型字符串
        CreatorRegistry & registry = Registry();///获取注册表指针

        ///确认有找到给定type的reator
        CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
            <<" (known types: " << LayerTypeListString() << ")";
        return registry[type](param);
        /// 根据layer name, 调用相应creator函数
    }

    ///返回layer type
    ///先创建vector类型的layer_types， 在遍历注册表然后把遍历到的值放进里容器中
    static vector<string> LayerTypeList() {
        CreatorRegistry & registry  = Registry(); /// 获取注册表指针
        vector<string> layer_types;
        ///遍历注册表，
        ///map调用begin() 函数取得第一个元素，
        for (typename CreatorRegistry::iterator iter = registry.begin();
        iter != registry.end(); ++iter)
        {layer_types.push_back(iter->first);
            ///vector类调用push_back将元素添加至最后

        }
        return layer_types;


    }
private:
    // Layer registry should never be instantiated - everything is done with its
    // static variables.
    LayerRegistry() {}

    static string LayerTypeListString() {
        vector<string> layer_types = LayerTypeList(); ///获取一个layer_types
        string layer_types_str; ///创建一个字符串类
        for (vector<string>::iterator iter = layer_types.begin();
             iter != layer_types.end(); ++iter) {
            if (iter != layer_types.begin()) {
                layer_types_str += ", ";
            }
            layer_types_str += *iter;
        }
        return layer_types_str;
    }


};


/// LayerRegisterer：Layer注册器,供后面的宏使用
template <typename Dtype>
class LayerRegisterer {
public:
    LayerRegisterer(const string& type, shared_ptr<Layer<Dtype>> (*creator)(const LayerParameter&))
    {
        LayerRegistry<Dtype>::AddCreator(type, creator);
        ///直接调用LayerRegistry静态成员函数
        ///向之前LayerRegistry的注册表中添加Creator
    }
};





#define REGISTER_LAYER_CREATOR(type, creator)                                   \
    static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
    static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)   \


/**
 * REGISTER_LAYER_CLASS执行两步骤
 * 1. 为每一个layer创建一个creator函数, 返回对象指针
 * 2. 将指定的Layer注册到全局注册表中
 * */

#define REGISTER_LAYER_CLASS(type)
    template <typename Dtype>                                                   \
    shared_ptr<Layer<Dtype>> Creator_##type##Layer(const LayerParameter & param)\
    {                                                                           \
        return shared_ptr<Layer<Dtype>> (new type##Layer<Dtype>(param));        \
    }                                                                           \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
} //namespace caffe

#endif //CAFFE_DETAIL_LAYER_FACTORY_HPP



