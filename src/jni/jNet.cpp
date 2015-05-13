#include <string>

#include "util.hpp"
#include "edu_h2r_jNet.h"

using std::string;
using caffe::Net;
using caffe::Blob;
using caffe::Layer;
using caffe::MemoryDataLayer;
using std::vector;
using std::cout;
using boost::shared_ptr;

JNIEXPORT jlong JNICALL Java_edu_h2r_jNet_createNet__Ljava_lang_String_2Ljava_lang_String_2(JNIEnv *env, jobject obj, jstring param_file, jstring pretrained_param_file) {

    FLAGS_minloglevel = 2;
    const char* c_param_file = env->GetStringUTFChars(param_file, NULL);
    const char* c_pretrained_param_file = env->GetStringUTFChars(pretrained_param_file, NULL);

    Net<float> *net = new Net<float>(string(c_param_file), caffe::TEST);
    net->CopyTrainedLayersFrom(string(c_pretrained_param_file));
    setInternalPtr<Net<float> >(env, obj, net);

    // Free up allocated space
    env->ReleaseStringUTFChars(param_file, c_param_file);
    env->ReleaseStringUTFChars(pretrained_param_file, c_pretrained_param_file);

    return reinterpret_cast<jlong>(net);
}

JNIEXPORT void JNICALL Java_edu_h2r_jNet_dispose(JNIEnv *env, jobject obj) {

    Net<float> *net = getInternalObject<Net<float> >(env, obj);
    setInternalPtr<Net<float> >(env, obj, NULL);
    delete net;
}

JNIEXPORT jfloatArray JNICALL Java_edu_h2r_jNet_forwardTo(JNIEnv *env, jobject obj,
                                        jfloatArray input, jstring to_layer_name) {
    float *c_input = env->GetFloatArrayElements(input, NULL);
    const char* c_to_layer_name = env->GetStringUTFChars(to_layer_name, NULL);

    // Normalize if necessary
    float scale = env->GetFloatField(obj, getObjField(env, obj, "inputScale", "F"));

    if (scale > EPSILON) {
        jsize len = env->GetArrayLength(input);
        for (int i = 0; i < len; i++) {
            c_input[i] *= scale;
        }
    }

    Net<float> *net = getInternalObject<Net<float> >(env, obj);

    // We're doing some of the convenience that
    // Net<Dtype>::Forward does for us by hand
    bool is_memory_data = (string("MemoryData").compare(net->layers()[0].get()->layer_param().type()) == 0);
    Blob<float> *data_input_blob;
    if(!is_memory_data){
        Blob<float> *net_input_blob = net->input_blobs()[0];
        data_input_blob = cloneWithNewData<float>(*net_input_blob, c_input);
        net_input_blob->CopyFrom(*data_input_blob);
    }


    int to = -1;
    string s = string(c_to_layer_name);
    std::vector<string> layer_names = net->layer_names();
    for (unsigned int i = 0; i < layer_names.size(); i++) {
        if (s.compare(layer_names[i]) == 0) {
            to = i;
            break;
        }
    }
    net->ForwardFromTo(0, to);

    // Get results
    string blob_name = string(net->layer_by_name(c_to_layer_name).get()->layer_param().top().Get(0));
    shared_ptr<Blob<float> > results = net->blob_by_name(blob_name);

    jfloatArray out = env->NewFloatArray(results->count());
    env->SetFloatArrayRegion(out, 0, results->count(), results->cpu_data());

    env->ReleaseStringUTFChars(to_layer_name, c_to_layer_name);
    if(!is_memory_data){
        delete data_input_blob;
    }
    delete c_input;

    return out;
}

JNIEXPORT jboolean JNICALL Java_edu_h2r_jNet_hasLayer(JNIEnv *env, jobject obj,
                                                        jstring layer_name) {
    const char* c_layer_name = env->GetStringUTFChars(layer_name, NULL);
    Net<float>* net = getInternalObject<Net<float> >(env, obj);

    const bool ret = net->has_layer(string(c_layer_name));
    env->ReleaseStringUTFChars(layer_name, c_layer_name);

    return ret;
}

JNIEXPORT jint JNICALL Java_edu_h2r_jNet_getNodeCount(JNIEnv *env, jobject obj,
                                                        jstring blob_name) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    const char* c_blob_name = env->GetStringUTFChars(blob_name, NULL);

    shared_ptr<Blob<float> > blob = net->blob_by_name(string(c_blob_name));

    env->ReleaseStringUTFChars(blob_name, c_blob_name);

    return reinterpret_cast<jint>(blob->count());
}

JNIEXPORT void JNICALL Java_edu_h2r_jNet_setMemoryDataLayer(JNIEnv *env, jobject obj, jstring layer_name,
                                                                    jfloatArray data, jfloatArray label){
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    const char* c_layer_name = env->GetStringUTFChars(layer_name, NULL);
    float* c_data = env->GetFloatArrayElements(data, NULL);
    float* c_label = env->GetFloatArrayElements(label, NULL);

    // Check if the layer with the specified name exists
    CHECK(net->has_layer(string(c_layer_name))) << "The layer with the specified name doesn't exist.";

    // Normalize if necessary
    float scale = env->GetFloatField(obj, getObjField(env, obj, "inputScale", "F"));

    jsize len_data = env->GetArrayLength(data);
    jsize len_label = env->GetArrayLength(label);
    float* cpy_data = new float[len_data];
    float* cpy_label = new float[len_label];

    if (scale > EPSILON) {
        for (int i = 0; i < len_data; i++) {
            cpy_data[i] = c_data[i] * scale;
        }
        for (int i = 0; i < len_label; i++) {
            cpy_label[i] = c_label[i] * scale;
        }
    } else {
        for (int i = 0; i < len_data; i++) {
            cpy_data[i] = c_data[i];
        }
        for (int i = 0; i < len_label; i++) {
            cpy_label[i] = c_label[i];
        }
    }
    delete c_data;
    delete c_label;

    Layer<float>* layer = net->layer_by_name(c_layer_name).get();
    MemoryDataLayer<float>* memoryDataLayer = reinterpret_cast<MemoryDataLayer<float>* >(layer);

    env->ReleaseStringUTFChars(layer_name, c_layer_name);

    memoryDataLayer->Reset(cpy_data, cpy_label, 1);
}

JNIEXPORT jobjectArray JNICALL Java_edu_h2r_jNet_getLayerNames(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);

    std::vector<string> layer_names = net->layer_names();

    jobjectArray java_layer_names = env->NewObjectArray(layer_names.size(),
                                                        env->FindClass("java/lang/String"),
                                                        NULL);
    for (unsigned int i = 0; i < layer_names.size(); i++)
        env->SetObjectArrayElement(java_layer_names, i,
                                    env->NewStringUTF(layer_names[i].c_str()));

    return java_layer_names;
}

JNIEXPORT jint JNICALL Java_edu_h2r_jNet_getInputHeight(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    return reinterpret_cast<jint>(net->input_blobs()[0]->height());
}

JNIEXPORT jint JNICALL Java_edu_h2r_jNet_getInputWidth(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    return reinterpret_cast<jint>(net->input_blobs()[0]->width());
}

// JNIEXPORT jint JNICALL Java_edu_h2r_jNet_getBlobNumber(JNIEnv *env, jobject obj,
//                                                         jstring blob_name) {
//     Net<float> *net = getInternalObject<Net<float> >(env, obj);

//     const char* c_blob_name = env->GetStringUTFChars(blob_name, NULL);
//     const string s = string(c_blob_name);


//     vector<string> blob_names = net->layer_names();
//     for (unsigned int i = 0; i < blob_names.size(); i++) {
//         if (s.compare(blob_names[i]) == 0) {
//             return i;
//         }
//     }

//     delete c_blob_name;

//     return -1;
// }
