#include <string>

#include "util.hpp"
#include "edu_h2r_jSolver.h"

using caffe::Solver;

JNIEXPORT jlong JNICALL Java_edu_h2r_jSolver_createSolver(JNIEnv *env,
                                            jobject obj, jstring solverFile) {
    FLAGS_minloglevel = 2;
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

    return (jlong) solver;
}

JNIEXPORT jlong JNICALL Java_edu_h2r_jSolver_getNetPointer(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    return (jlong)solver->net();
}

JNIEXPORT void JNICALL Java_edu_h2r_jSolver_train(JNIEnv *env, jobject obj) {
    Solver<float> *solver = getInternalObject<Solver<float> >(env, obj);
    solver->Solve();
}

