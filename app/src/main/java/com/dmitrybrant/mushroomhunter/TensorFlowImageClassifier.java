package com.dmitrybrant.mushroomhunter;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "ImageClassifier";

    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    private int inputSize;
    private int imageMean;
    private float imageStd;

    private final Vector<String> labels = new Vector<>();
    private int[] intValues;
    private ByteBuffer imgData;
    private float[][] outputs;

    private Interpreter interpreter;

    private TensorFlowImageClassifier() {}

    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        String actualLabelFilename = labelFilename.contains("android_asset/")
                ? labelFilename.split("file:///android_asset/")[1] : labelFilename;
        Log.i(TAG, "Reading labels from: " + actualLabelFilename);
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(assetManager.open(actualLabelFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        String actualModelFilename = modelFilename.contains("android_asset/")
                ? modelFilename.split("file:///android_asset/")[1] : modelFilename;
        try {
            c.interpreter = new Interpreter(loadModelFile(assetManager, actualModelFilename));
        } catch (IOException e) {
            throw new RuntimeException("Problem loading model file!", e);
        }

        int numClasses = c.interpreter.getOutputTensor(0).shape()[1];
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // 1 image * inputSize * inputSize * 3 channels * 4 bytes per float
        c.imgData = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4);
        c.imgData.order(ByteOrder.nativeOrder());
        c.intValues = new int[inputSize * inputSize];
        c.outputs = new float[1][numClasses];

        return c;
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixel = intValues[i * inputSize + j];
                imgData.putFloat((((pixel >> 16) & 0xFF) - imageMean) / imageStd);
                imgData.putFloat((((pixel >> 8) & 0xFF) - imageMean) / imageStd);
                imgData.putFloat(((pixel & 0xFF) - imageMean) / imageStd);
            }
        }
        Trace.endSection();

        Trace.beginSection("run");
        interpreter.run(imgData, outputs);
        Trace.endSection();

        PriorityQueue<Recognition> pq = new PriorityQueue<>(3,
                (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));
        for (int i = 0; i < outputs[0].length; ++i) {
            if (outputs[0][i] > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labels.size() > i ? labels.get(i) : "unknown", outputs[0][i], null));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection();
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
