This is a set of Java bindings to the [Caffe](http://caffe.berkeleyvision.org/)
deep learning framework.

# Prerequisites

1. Caffe - [installation instructions](http://caffe.berkeleyvision.org/installation.html)
2. JNI - this is already installed on most Linux operating systems

# Installation

1. Download this repository and place it within your caffe home directory
2. Run ``make java`` or ``make`` inside the newly downloaded directory

This will generate a ``.jar`` and ``.so`` file in the ``lib/`` directory.

# Usage
In order to use the Java bindings you must:

1. Add the ``caffe_jni.jar`` file to your project's dependencies
2. Make sure ``/usr/local/cuda/lib64`` and ``$(CAFFE_HOME)/build/lib`` are both
    in the ``LD_LIBRARY_PATH`` environment variable.