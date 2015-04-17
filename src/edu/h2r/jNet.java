package edu.h2r;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;

/**
 * TODO: 1) Add method to query number of nodes in layer
 * TODO: 2) Add c++ error handling
 * TODO: 3) Optimize image operations
 * TODO: 4) Have an internal thread create & destroy c++ stuff
 *
 */
public class jNet implements Disposable {

    private float inputScale;
    private long internalPtr;
    private String solverFile;

    /**
     * @param model_file the simplified model file (aka the deploy file)
     * @param pretrained_param_file a trained .caffemodel file
     * @param inputScale a float used to multiply inputs and scale them
     */
    public jNet(String model_file, String pretrained_param_file, float inputScale) {
        internalPtr = createNet(model_file, pretrained_param_file);
        this.inputScale = inputScale;
    }

    /** Create and initialize a net from a solver file
     * @param solver_file the solver.prototxt file
     */
    public jNet(String solver_file) {
        this.internalPtr = createNet(solver_file);
        this.solverFile = solver_file;
        this.inputScale = -1;

        // Parse the network definition file name from the solver file
        String basePath = solver_file.split("/(?=[^/]+$)")[0];
        String networkFile = null;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(solver_file)));
            String line;
            while((line = br.readLine()) != null)
                if(line.trim().startsWith("net:")){
                    String tmp = line.trim().substring(line.trim().indexOf("net:") + 4).trim();
                    networkFile = tmp.substring(1, tmp.length() - 1);
                    if(!networkFile.startsWith("/"))
                        networkFile = basePath + "/" + networkFile;
                    break;
                }
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Parse the input scale from the network definition file
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(networkFile)));
            String line;
            while((line = br.readLine()) != null)
                if(line.trim().startsWith("scale:")){
                    String tmp = line.trim().substring(line.trim().indexOf("scale:") + 6).trim();
                    this.inputScale = Float.valueOf(tmp);
                    break;
                }
            br.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (this.inputScale == -1)
            throw new IllegalArgumentException("Couldn't parse the 'scale' parameter from file " + networkFile +
                    " (parsed from file " + this.solverFile + ")");
    }

    protected jNet(long internalPtr, float inputScale) {
        this.internalPtr = internalPtr;
        this.inputScale = inputScale;
    }

    /**
     * Performs a full forward pass through the network and returns its output.
     * @param image the image to use as input
     * @return the output of the network
     * @throws IllegalArgumentException
     */
    public float[] forward(BufferedImage image) throws IllegalArgumentException {
        if (image.getWidth() != getInputWidth() || image.getHeight() != getInputHeight())
            throw new IllegalArgumentException("Image does not have correct dimensions");

        String[] layerNames = getLayerNames();
        return forwardTo(imageToArray(image), layerNames[layerNames.length - 1]);
    }

    /**
     * Runs a forward pass through the network and returns the output at a specific layer.
     * The forward pass is done only up to the specified layer.
     * @param image the image to use as input
     * @param toLayerName the layer whose output will be returned
     * @return the output of layer toLayerName
     * @throws IllegalArgumentException
     */
    public float[] forwardTo(BufferedImage image, String toLayerName) throws IllegalArgumentException {
        if (image.getWidth() != getInputWidth() || image.getHeight() != getInputHeight())
            throw new IllegalArgumentException("Image does not have correct dimensions");

        if (!hasLayer(toLayerName))
            throw new IllegalArgumentException("Layer does not exist");

        return forwardTo(imageToArray(image), toLayerName);
    }

    /**
     * Converts a {@link java.awt.image.BufferedImage} to a float array.
     * @param image the image to convert
     * @return a float array whose values are the pixel values of the image
     */
    public static float[] imageToArray(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();

        float[] pixels = new float[ h * w];
        image.getData().getPixels(0, 0, w, h, pixels);
        return pixels;
    }

    /**
     * Creates a {@link java.awt.image.BufferedImage} from an array of pixels.
     * @param pixels the pixels used to create the image
     * @param width the image's width
     * @param height the image's height
     * @return a {@link java.awt.image.BufferedImage}
     */
    public static BufferedImage imageFromArray(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = image.getRaster();
        raster.setPixels(0, 0, width, height, pixels);
        return image;
    }

    /**
     * Converts gray-scale pixels to RGB values.
     * @param grayPixels float array of gray-scale pixels
     * @return
     */
    public static int[] grayToRGB(float[] grayPixels) {
        int[] rgbPixels = new int[grayPixels.length];
        for (int i = 0; i < grayPixels.length; i++) {
            int grayPixel = (int) (grayPixels[i] * 255f);
            rgbPixels[i] = -16777216;  // alpha channel
            rgbPixels[i] += (grayPixel & 0xFF);
            rgbPixels[i] += (grayPixel & 0xFF) << 8;
            rgbPixels[i] += (grayPixel & 0xFF) << 16;
        }
        return rgbPixels;
    }

    /**
     * Converts gray-scale pixels to RGB values.
     * @param grayPixels integer array of gray-scale pixels
     * @return
     */
    public static int[] grayToRGB(int[] grayPixels) {
        int[] rgbPixels = new int[grayPixels.length];
        for (int i = 0; i < grayPixels.length; i++) {
            rgbPixels[i] = -16777216;  // alpha channel
            rgbPixels[i] += (grayPixels[i] & 0xFF);
            rgbPixels[i] += (grayPixels[i] & 0xFF) << 8;
            rgbPixels[i] += (grayPixels[i] & 0xFF) << 16;
        }
        return rgbPixels;
    }

    /**
     * The following methods are implemented in C++.
     * See edu_h2r_JNet.h and _net.cpp for their C++ definitions and implementations.
     */

    /**
     * Instantiates a Caffe neural network from the model_file and sets its weights using the pretrained_param_file.
     * The {@link jNet} class is a wrapper around the underlying Caffe neural network, so store a pointer to
     * it in the internalPtr parameter.
     * @param model_file a Caffe model definition file
     * @param pretrained_param_file a .caffemodel file that contains weights for the Caffe model we are using
     * @return a pointer to the underlying Caffe neural network
     */
    private native long createNet(String model_file, String pretrained_param_file);

    /**
     * Instantiates a Caffe neural network from the solver_file.
     * The {@link jNet} class is a wrapper around the underlying Caffe neural network, so store a pointer to
     * it in the internalPtr parameter.
     * @param solver_file a Caffe model definition file
     * @return a pointer to the underlying Caffe neural network
     */
    private native long createNet(String solver_file);

    /**
     * Deletes the underlying Caffe neural network
     */
    public native void dispose();

    /**
     * A wrapper around the Caffe neural network's forwardTo method.
     * @param input the input data
     * @param toLayerName the name of the layer whose output we wish to retrieve
     * @return the output from the layer whose name is toLayerName
     */
    public native float[] forwardTo(float[] input, String toLayerName);

    /**
     * @return the names of all the layers in the neural network
     */
    public native String[] getLayerNames();

    /**
     * Gets the number of nodes a layer has.
     * @param layerName
     * @return
     */
    public native int getNodeCount(String layerName);

    /**
     * @return the input height that the Caffe neural network is expecting.
     */
    public native int getInputHeight();

    /**
     * @return the input width that the Caffe neural network is expecting.
     */
    public native int getInputWidth();

    /**
     * Checks to see if the neural network has a layer whose name is layerName.
     * @param layerName
     * @return
     */
    private native boolean hasLayer(String layerName);

    static {
        File jar = new File(jNet.class.getProtectionDomain().getCodeSource().getLocation().getPath());
        System.load(jar.getParentFile().toURI().resolve("libcaffe_jni.so").getPath());
    }
}
