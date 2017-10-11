/*
 * Copyright 2017 Dmitry Brant
 *
 * Adapted from sample classifier app from the TensorFlow repo.
 */
package com.dmitrybrant.mushroomhunter;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.widget.TextView;

import org.tensorflow.demo.env.ImageUtils;

import java.util.List;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {

    // These are the settings for the original v1 Inception model. If you want to
    // use a model that's been produced from the TensorFlow for Poets codelab,
    // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
    // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
    // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
    // the ones you produced.
    //
    // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
    // model first:
    //
    // python strip_unused.py \
    // --input_graph=<retrained-pb-file> \
    // --output_graph=<your-stripped-pb-file> \
    // --input_node_names="Mul" \
    // --output_node_names="final_result" \
    // --input_binary=true

    /*
    // original Inception v1 model from sample
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    */

    /*
    // retrained Inception V3
    private static final int INPUT_SIZE = 299;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String INPUT_NAME = "Mul";
    private static final String OUTPUT_NAME = "final_result";
    */

    // retrained MobileNet v1
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "final_result";

    private MinStreakResult streakResult = new MinStreakResult(3);

    private static final String MODEL_FILE = "file:///android_asset/mushrooms.pb";
    private static final String LABEL_FILE = "file:///android_asset/output_labels.txt";

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final boolean MAINTAIN_ASPECT = true;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private Classifier classifier;

    private Integer sensorOrientation;

    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private Bitmap cropCopyBitmap;

    private boolean computing = false;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private TextView resultsView;

    private TextView borderedText;

    private long lastProcessingTimeMs;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        new AlertDialog.Builder(this)
                .setTitle(R.string.disclaimer_title)
                .setMessage(R.string.disclaimer_body)
                .setCancelable(false)
                .setPositiveButton(android.R.string.ok, null)
                .create()
                .show();
    }
    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        borderedText = new TextView(this);
        borderedText.setTypeface(Typeface.MONOSPACE);

        classifier = TensorFlowImageClassifier.create(
                getAssets(),
                MODEL_FILE,
                LABEL_FILE,
                INPUT_SIZE,
                IMAGE_MEAN,
                IMAGE_STD,
                INPUT_NAME,
                OUTPUT_NAME);

        resultsView = findViewById(R.id.results);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        //LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        sensorOrientation = rotation + screenOrientation;

        //LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                INPUT_SIZE, INPUT_SIZE,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

        addCallback(new OverlayView.DrawCallback() {
            @Override
            public void drawCallback(final Canvas canvas) {
                renderDebug(canvas);
            }
        });
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image image = null;

        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (computing) {
                image.close();
                return;
            }
            computing = true;

            Trace.beginSection("imageAvailable");

            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            image.close();
        } catch (final Exception e) {
            e.printStackTrace();
            if (image != null) {
                image.close();
            }
            Trace.endSection();
            return;
        }

        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(new Runnable() {
            @Override
            public void run() {
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                displayResults(results);
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

                requestRender();
                computing = false;
            }
        });

        Trace.endSection();
    }

    @Override
    public void onSetDebug(boolean debug) {
        classifier.enableStatLogging(debug);
    }

    private void displayResults(List<Classifier.Recognition> results) {
        float confidenceThreshold = 0.75f;
        if (results.size() > 0 && results.get(0).getConfidence() >= confidenceThreshold) {
            streakResult.add(results.get(0));
        } else {
            streakResult.add(null);
        }

        Classifier.Recognition result = streakResult.get();
        final String resultStr = result == null ? ""
                : (result.getTitle() + " (" + (int) (result.getConfidence() * 100f) + "%)");

        resultsView.post(new Runnable() {
            @Override
            public void run() {
                if (resultStr.toLowerCase().contains("not")) {
                    resultsView.setText("");
                    return;
                }
                resultsView.setText(resultStr);
                if (resultStr.toLowerCase().contains("fly") || resultStr.toLowerCase().contains("destroy")) {
                    resultsView.setBackgroundColor(ContextCompat.getColor(ClassifierActivity.this, R.color.poisonousBackground));
                } else {
                    resultsView.setBackgroundColor(ContextCompat.getColor(ClassifierActivity.this, R.color.edibleBackground));
                }
            }
        });
    }

    private void renderDebug(final Canvas canvas) {
        if (!isDebug()) {
            return;
        }
        final Bitmap copy = cropCopyBitmap;
        if (copy != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                    canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            StringBuilder lines = new StringBuilder();
            if (classifier != null) {
                String statString = classifier.getStatString();
                String[] statLines = statString.split("\n");
                for (String line : statLines) {
                    lines.append(line);
                    lines.append("\n");
                }
            }

            lines.append("Frame: " + previewWidth + "x" + previewHeight);
            lines.append("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.append("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.append("Rotation: " + sensorOrientation);
            lines.append("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.setText(lines);
        }
    }
}
