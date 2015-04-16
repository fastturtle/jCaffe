PROJECT := caffe

JAVA_JDK := $$(readlink -f /usr/bin/javac | sed "s:bin/javac::")

JAVA$(PROJECT)_SRC := src/jni/*.cpp
JAVA$(PROJECT)_HXX_SRC := src/jni/*.hpp
JAVA$(PROJECT)_SO := lib/lib$(PROJECT).so

LINKFLAGS := -pthread -fPIC -Wall -I../.build_release/src -I../caffe/src -I../include -I/usr/local/cuda/include
JAVA_FLAGS := -I$(JAVA_JDK)/include -I$(JAVA_JDK)include/linux


java: $(JAVA$(PROJECT)_SO)

$(JAVA$(PROJECT)_SO): $(JAVA$(PROJECT)_SRC)
	mkdir -p lib
	$(CXX) -shared -o $@ $(JAVA$(PROJECT)_SRC) $(LINKFLAGS) $(JAVA_FLAGS) -L../build/lib -lcaffe
	@ echo
	mkdir -p out
	javac -d out src/edu/h2r/JNet.java src/edu/h2r/Disposable.java
	jar cf lib/caffe_jni.jar out/edu/* lib/libcaffe.so
