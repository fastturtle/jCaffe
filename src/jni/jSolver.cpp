#include <string>

//#include "jni_exception.h"
#include "util.hpp"
#include "edu_h2r_jSolver.h"

using caffe::Solver;
using caffe::LayerParameter;


JNIEXPORT jlong JNICALL Java_edu_h2r_jSolver_createSolver(JNIEnv *env,
                                            jobject obj, jstring solverFile) {
    //SAVE_PGM_STATE();
    //JNI_ASSERT(0, "Coucou");
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

JNIEXPORT void JNICALL Java_edu_h2r_jSolver__1dispose(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    setInternalPtr<Solver<float> >(env, obj, NULL);
    delete solver;
}

