package edu.h2r;

import com.sun.javaws.exceptions.InvalidArgumentException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by gabe on 2/19/15.
 */
public class Test {
    public static void main(String[] args) throws IOException, InvalidArgumentException {
        JNet net = new JNet("/home/gabe/deeprl-autoencoder/corridor/deploy.prototxt", "/home/gabe/deeprl-autoencoder/corridor/snapshots/autoencoder_iter_150000.caffemodel", 1.0f / 255.0f );

        for (int imgNum = 0; imgNum < 10; imgNum++) {
            BufferedImage image = ImageIO.read(new File("/home/gabe/deeprl-autoencoder/corridor/images/10states/" + imgNum + ".jpg"));
            float[] output = net.forwardTo(image, "encode1neuron");
            for (int i = 0; i < output.length; i++) {
                System.out.printf("%.4f ", output[i]);
            }
            System.out.printf("\n");
//            System.out.println(output.length);
//            System.out.printf("Image %d: ", imgNum);
//            int n = 0;
//            int[] pixels = new int[output.length];
//            for (int i = 0; i < output.length; i++) {
//                int grayByte = (int) (output[i] * 255f) & 0xFF;
//                pixels[i] = -16777216;
//                pixels[i] += grayByte;
//                pixels[i] +=  (grayByte << 8);
//                pixels[i] += (grayByte << 16);
//            }
//            int[] pixels = JNet.grayToRGB(output);
//            BufferedImage reconstructed = JNet.imageFromArray(pixels, 80, 20);
//            ImageIO.write(reconstructed,"jpg", new File(imgNum + ".jpg"));

        }
        net.dispose();

    }
}
