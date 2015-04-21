/********************************************************************************
File : jni_exception.cpp
Author:Jafar K (jafarmlp@googlemail.com)
Date/version : 02/02/07
********************************************************************************/
#include <jni.h>

#include "jni_exception.h"

JavaVM *cached_jvm;                                   // A pointer to the VM from which
                                                      //we can get the JNIEnv for doing callbacks:
char  g_azErrorMessage[ERROR_MESSAGE_LENGTH] = {0};   // Error message to Java 
/******************************************************************************
JNI_OnLoad will be called when the jvm loads this library
******************************************************************************/
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv *env;
    cached_jvm = jvm;  /* cache the JavaVM pointer */

    if ( jvm->GetEnv((void **)&env, JNI_VERSION_1_2)) {
       printf("JNI version is not supported");
       return JNI_ERR; /* JNI version not supported */
    }
    return JNI_VERSION_1_2;
}
/******************************************************************************
Restore the saved state.
Param       :const char* pzFile,
             int iLine,
             const char* pzMessage
Return      :
******************************************************************************/
void RestoreProgramState(const char* pzFile, int iLine,const char* pzMessage) {
   //Copy the error message to the global array.
   sprintf(g_azErrorMessage,"JNIException ! \n \
      File \t\t:  %s \n \
      Line number \t\t: %d \n \
      Reason for Exception\t: %s ",pzFile,iLine,pzMessage);
   //Restore the saved/safe state.
   // RESTORE_SAFE_STATE();
   printf("%p\n", g_sJmpbuf);
   longjmp(g_sJmpbuf, 1);
   // ThrowJNIException(pzFile, iLine, pzMessage);
}
/******************************************************************************
throws the exception to java
Param       :const char* pzFile,
             int iLine,
             const char* pzMessage
Return      :
******************************************************************************/
void ThrowJNIException(const char* pzFile, int iLine,const char* pzMessage)
{
   //Check for null parameter
   if(pzFile != NULL && pzMessage != NULL && iLine != 0)
      sprintf(g_azErrorMessage,"JNIException ! \n \
      File \t\t:  %s \n \
      Line number \t\t: %d \n \
      Reason for Exception\t: %s ",pzFile,iLine,pzMessage);
   jclass    tClass        = NULL;
   //Findout the exception handling class
   JNIEnv *env;
   cached_jvm->AttachCurrentThread( (void **)&env, NULL );

   if( env == NULL) {
      printf("Invalid null pointer in ThrowJNIException " );
      return;
   }

   tClass = env->FindClass(JNI_EXCEPTION_CLASS_NAME);
   if (tClass == NULL) {
     printf("Not found %s",JNI_EXCEPTION_CLASS_NAME);
     return;
   }
   //Throw the excption with error info
   env->ThrowNew(tClass,g_azErrorMessage);
   env->DeleteLocalRef(tClass);
}