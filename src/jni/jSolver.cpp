#include <string>
// #include <setjmp.h>

//#include "jni_exception.h"
#include "util.hpp"
#include "edu_h2r_jSolver.h"

using caffe::Solver;
using caffe::Net;
using caffe::LayerParameter;


JNIEXPORT jlong JNICALL Java_edu_h2r_jSolver_createSolver(JNIEnv *env,
                                            jobject obj, jstring solverFile) {
    const char* cSolverFile = env->GetStringUTFChars(solverFile, NULL);

    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(cSolverFile, &solver_param);

    int gpu = -1;
    if (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU)
        gpu = solver_param.device_id();
    
    if (gpu >= 0) {
        caffe::Caffe::SetDevice(gpu);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } else {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }

    Solver<float> *solver = caffe::GetSolver<float>(solver_param);
    JNI_ASSERT(solver != NULL, "caffe::GetSolver returned a null pointer");

    setInternalPtr<Solver<float> >(env, obj, solver);

    env->ReleaseStringUTFChars(solverFile, cSolverFile);

    // Send the input scale back to java
    caffe::NetParameter net_param;
    caffe::ReadProtoFromTextFileOrDie(solver_param.net(), &net_param);
    const LayerParameter& layer_param = net_param.layer(0);
    env->SetFloatField(obj, getObjField(env, obj, "inputScale", "F"), layer_param.transform_param().scale());

    return reinterpret_cast<jlong>(solver);
}

JNIEXPORT jlong JNICALL Java_edu_h2r_jSolver_getNetPointer(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    return reinterpret_cast<jlong>(solver->net().get());
}

JNIEXPORT void JNICALL Java_edu_h2r_jSolver_train(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    solver->Solve();
}

JNIEXPORT void JNICALL Java_edu_h2r_jSolver_trainOneStep(JNIEnv *env, jobject obj){
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    Net<float> *net_ = solver->net().get();

    LOG(INFO) << "Solving " << net_->name();

    solver->Step(1);

    float loss;
    solver->iter();
    net_->ForwardPrefilled(&loss);

    LOG(INFO) << "Optimization Done.";
}
JNIEXPORT void JNICALL Java_edu_h2r_jSolver_setLogLevel(JNIEnv *env, jobject obj, jint log_level){
    const int c_log_level = reinterpret_cast<int>(log_level);
    FLAGS_minloglevel = c_log_level;
}

JNIEXPORT void JNICALL Java_edu_h2r_jSolver__1dispose(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    setInternalPtr<Solver<float> >(env, obj, NULL);
    delete solver;
}

