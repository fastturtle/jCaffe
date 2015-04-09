#include <string>
// #include <cmath>

#include "util.hpp"
#include "edu_h2r_JNet.h"
#include "caffe/caffe.hpp"

using std::string;
using caffe::Net;
using caffe::Blob;
using std::vector;
using std::cout;
using boost::shared_ptr;

JNIEXPORT jlong JNICALL Java_edu_h2r_JNet_createNet(JNIEnv *env, jobject obj, jstring param_file, jstring pretrained_param_file) {
    
    FLAGS_minloglevel = 2;
    const char* c_param_file = env->GetStringUTFChars(param_file, NULL);
    const char* c_pretrained_param_file = env->GetStringUTFChars(pretrained_param_file, NULL);

    Net<float> *net = new Net<float>(string(c_param_file), caffe::TEST);
    net->CopyTrainedLayersFrom(string(c_pretrained_param_file));
    setInternalPtr<Net<float> >(env, obj, net);

    // Free up allocated space
    env->ReleaseStringUTFChars(param_file, c_param_file);
    env->ReleaseStringUTFChars(pretrained_param_file, c_pretrained_param_file);

    // std::vector<string> layer_names = net->layer_names();
    // std::vector<shared_ptr<caffe::Layer<float> > > layers = net->layers();
    // for (int i = 0; i < layer_names.size(); i++) {
        // printf("Layer[%s] %d: %s blobs: %lu\n", layers[i]->type_name().c_str(), i, layer_names[i].c_str(), layers[i]->blobs().size());
        // cout << "Layer " << i << ": " << layer_names[i] << "blobs: " << layers[i]->blobs().size() << "\n";
    // }

    // std::vector<string> blob_names = net->blob_names();
    // std::vector<shared_ptr<Blob<float> > > blobs = net->blobs();
    // for (int i = 0; i < blob_names.size(); i++) {
    //     printf("Blob %d: %s\n", i, blob_names[i].c_str());        
    // }

    return (jlong) net;
}

JNIEXPORT void JNICALL Java_edu_h2r_JNet_dispose(JNIEnv *env, jobject obj) {

    Net<float> *net = getInternalObject<Net<float> >(env, obj);
    setInternalPtr<Net<float> >(env, obj, NULL);
    delete net;    
}


// JNIEXPORT jfloatArray JNICALL Java_edu_h2r_JNet_forward(JNIEnv *env, jobject obj,
//                                         jfloatArray input) {
//     float *c_input = env->GetFloatArrayElements(input, NULL);

//     jsize len = env->GetArrayLength(input);
//     for (int i = 0; i < len; i++)
//         // cout << c_input[i] << "\n";
//         c_input[i] /= 255.0;


//     Net<float> *net = getInternalObject<Net<float> >(env, obj);

//     Blob<float>* data_input_blob = cloneWithNewData(*(net->input_blobs()[0]), c_input);

//     vector<Blob<float>* > inputs;
//     inputs.push_back(data_input_blob);

//     float loss;
//     vector<Blob<float>* > results = net->Forward(inputs, &loss);

//     jfloatArray out = env->NewFloatArray(results[0]->count());
//     env->SetFloatArrayRegion(out, 0, results[0]->count(), results[0]->cpu_data());

//     delete data_input_blob;
//     delete c_input;

//     return out;
// }

JNIEXPORT jfloatArray JNICALL Java_edu_h2r_JNet_forwardTo(JNIEnv *env, jobject obj,
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
    Blob<float> *net_input_blob = net->input_blobs()[0];
    Blob<float> *data_input_blob = cloneWithNewData<float>(*net_input_blob, c_input);
    net_input_blob->CopyFrom(*data_input_blob);

    // vector<Blob<float>* > inputs;
    // inputs.push_back(data_input_blob);

    // net->Forward(inputs, NULL);

    int to = -1;
    string s = string(c_to_layer_name);
    std::vector<string> layer_names = net->layer_names();
    for (int i = 0; i < layer_names.size(); i++) {
        if (s.compare(layer_names[i]) == 0) {
            to = i;
            break;
        }
    }
    net->ForwardFromTo(0, to);

    // Get results
    shared_ptr<Blob<float> > results = net->blob_by_name(string(c_to_layer_name));

    jfloatArray out = env->NewFloatArray(results->count());
    env->SetFloatArrayRegion(out, 0, results->count(), results->cpu_data());

    env->ReleaseStringUTFChars(to_layer_name, c_to_layer_name);
    delete data_input_blob;
    delete c_input;

    return out;
}

JNIEXPORT jboolean JNICALL Java_edu_h2r_JNet_hasLayer(JNIEnv *env, jobject obj,
                                                        jstring layer_name) {
    const char* c_layer_name = env->GetStringUTFChars(layer_name, NULL);
    Net<float>* net = getInternalObject<Net<float> >(env, obj);

    const bool ret = net->has_layer(string(c_layer_name));
    env->ReleaseStringUTFChars(layer_name, c_layer_name);

    return ret;
}

JNIEXPORT jint JNICALL Java_edu_h2r_JNet_getNodeCount(JNIEnv *env, jobject obj,
                                                        jstring blob_name) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    const char* c_blob_name = env->GetStringUTFChars(blob_name, NULL);

    shared_ptr<Blob<float> > blob = net->blob_by_name(string(c_blob_name));

    env->ReleaseStringUTFChars(blob_name, c_blob_name);

    return blob->count();
}

JNIEXPORT jobjectArray JNICALL Java_edu_h2r_JNet_getLayerNames(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);

    std::vector<string> layer_names = net->layer_names();

    jobjectArray java_layer_names = env->NewObjectArray(layer_names.size(),
                                                        env->FindClass("java/lang/String"),
                                                        NULL);
    for (int i = 0; i < layer_names.size(); i++)
        env->SetObjectArrayElement(java_layer_names, i,
                                    env->NewStringUTF(layer_names[i].c_str()));

    return java_layer_names;
}

JNIEXPORT jint JNICALL Java_edu_h2r_JNet_getInputHeight(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    return net->input_blobs()[0]->height();
}

JNIEXPORT jint JNICALL Java_edu_h2r_JNet_getInputWidth(JNIEnv *env, jobject obj) {
    Net<float>* net = getInternalObject<Net<float> >(env, obj);
    return net->input_blobs()[0]->width();
}

JNIEXPORT jint JNICALL Java_edu_h2r_JNet_getBlobNumber(JNIEnv *env, jobject obj,
                                                        jstring blob_name) {
    Net<float> *net = getInternalObject<Net<float> >(env, obj);

    const char* c_blob_name = env->GetStringUTFChars(blob_name, NULL);
    const string s = string(c_blob_name);


    vector<string> blob_names = net->layer_names();
    for (int i = 0; i < blob_names.size(); i++) {
        if (s.compare(blob_names[i]) == 0) {
            // printf("Returning Layer[%d]: %s\n", i, c_blob_name);
            return i;
        }
    }

    delete c_blob_name;

    return -1;
}
