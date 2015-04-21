package edu.h2r;

import java.io.File;

/**
 * @author jafarmlp@googlemail.com
 */
public class JniException extends RuntimeException {
    private static final long serialVersionUID = -83699176624135323L;

    /**
     * Constructs a JNI exception with given message.
     * 
     * @param message
     *            Message to display
     */
    public JniException(String message) {
        super(message);
    }

    static {
        File jar = new File(jNet.class.getProtectionDomain().getCodeSource().getLocation().getPath());
        System.load(jar.getParentFile().toURI().resolve("libcaffe_jni.so").getPath());
    }

}