PROJECT := caffe

JAVA_JDK := $$(readlink -f /usr/bin/javac | sed "s:bin/javac::")

JAVA$(PROJECT)_SRC := src/jni/*.cpp
JAVA$(PROJECT)_HXX_SRC := src/jni/*.hpp
JAVA$(PROJECT)_SO := lib/libcaffe_jni.so
JAVA$(PROJECT)_JAR := lib/caffe_jni.jar

LINKFLAGS := -pthread -fPIC -Wall -I../.build_release/src -I../caffe/src \
				-I../include -I/usr/local/cuda/include -I./include
JAVA_FLAGS := -I$(JAVA_JDK)/include -I$(JAVA_JDK)include/linux


$(JAVA$(PROJECT)_SO): $(JAVA$(PROJECT)_SRC)
	$(CXX) -shared -o $@ $(JAVA$(PROJECT)_SRC) $(LINKFLAGS) $(JAVA_FLAGS) -L../build/lib -lcaffe
	@ echo
	mkdir -p lib
	javac -d . src/edu/h2r/jNet.java src/edu/h2r/jSolver.java src/edu/h2r/Disposable.java
	cp src/edu/h2r/jNet.java edu/h2r/
	cp src/edu/h2r/jSolver.java edu/h2r/
	cp src/edu/h2r/Disposable.java edu/h2r/
	jar cf $(JAVA$(PROJECT)_JAR) edu/*
	rm -r edu

clean:
	@- $(RM) $(JAVA$(PROJECT)_SO)
	@- $(RM) $(JAVA$(PROJECT)_JAR)
