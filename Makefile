PROJECT := caffe

JAVA_JDK := $$(readlink -f /usr/bin/javac | sed "s:bin/javac::")

JAVA$(PROJECT)_SRC := src/jni/*.cpp
JAVA$(PROJECT)_HXX_SRC := src/jni/*.hpp
JAVA$(PROJECT)_SO := lib/libcaffe_jni.so
JAVA$(PROJECT)_JAR := lib/caffe_jni.jar

LINKFLAGS := -pthread -fPIC -Wall -I../.build_release/src -I../caffe/src \
				-I../include -I/usr/local/cuda/include
JAVA_FLAGS := -I$(JAVA_JDK)/include -I$(JAVA_JDK)include/linux


java: $(JAVA$(PROJECT)_SO)

$(JAVA$(PROJECT)_SO): $(JAVA$(PROJECT)_SRC)
	mkdir -p lib
	$(CXX) -shared -o $@ $(JAVA$(PROJECT)_SRC) $(LINKFLAGS) $(JAVA_FLAGS) -L../build/lib -lcaffe
	@ echo
	javac -d . src/edu/h2r/JNet.java src/edu/h2r/Disposable.java
	jar cf $(JAVA$(PROJECT)_JAR) edu/*
	rm -r edu

clean:
	@- $(RM) $(JAVA$(PROJECT)_SO)
	@- $(RM) $(JAVA$(PROJECT)_JAR)
