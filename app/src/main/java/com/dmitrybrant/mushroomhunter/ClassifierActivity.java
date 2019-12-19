/*
 * Copyright 2018 Dmitry Brant
 *
 * Adapted from sample classifier app from the TensorFlow repo.
 */
package com.dmitrybrant.mushroomhunter;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.Trace;
import androidx.annotation.DrawableRes;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.cardview.widget.CardView;
import android.util.Size;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.dmitrybrant.mushroomhunter.util.ImageUtils;

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

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 320);

    private Classifier classifier;

    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private boolean computing = false;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    @Nullable private Classifier.Recognition currentResult;

    private CardView resultsCard;
    private TextView resultsTitleView;
    private TextView resultsBinomialView;
    private TextView resultsConfidenceView;
    private ImageView resultsImage;

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
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_about:
                new AlertDialog.Builder(this)
                        .setTitle(R.string.app_name)
                        .setMessage(R.string.about_message)
                        .setPositiveButton(android.R.string.ok, null)
                        .show();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        classifier = TensorFlowImageClassifier.create(
                getAssets(),
                MODEL_FILE,
                LABEL_FILE,
                INPUT_SIZE,
                IMAGE_MEAN,
                IMAGE_STD,
                INPUT_NAME,
                OUTPUT_NAME);

        resultsCard = findViewById(R.id.result_card);
        resultsTitleView = findViewById(R.id.result_title_text);
        resultsBinomialView = findViewById(R.id.result_binomial_text);
        resultsConfidenceView = findViewById(R.id.result_confidence_text);
        resultsImage = findViewById(R.id.result_preview_image);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();



        View tempView = findViewById(R.id.result_info_button);
        tempView.setOnClickListener(v -> {
            if (currentResult != null) {
                Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://en.m.wikipedia.org/wiki/" + currentResult.getTitle()));
                startActivity(browserIntent);
            }
        });



        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        //LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        int sensorOrientation = rotation + screenOrientation;

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

        runInBackground(() -> {
            computing = true;
            try {
                final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                displayResults(results);
            } finally {
                computing = false;
            }
        });

        Trace.endSection();
    }

    private void displayResults(List<Classifier.Recognition> results) {
        float confidenceThreshold = 0.75f;
        if (results.size() > 0 && results.get(0).getConfidence() >= confidenceThreshold) {
            streakResult.add(results.get(0));
        } else {
            streakResult.add(null);
        }

        currentResult = streakResult.get();

        resultsTitleView.post(() -> {

            boolean resultExists = currentResult != null;
            if (currentResult != null && currentResult.getTitle().toLowerCase().contains("not")) {
                resultExists = false;
            }

            if (resultExists) {

                String resultLower = currentResult.getTitle().toLowerCase();

                resultsTitleView.setText(getCommonNameForBinomial(resultLower));
                resultsBinomialView.setText(currentResult.getTitle());
                resultsConfidenceView.setText(((int) (currentResult.getConfidence() * 100f)) + "%");

                Glide.with(this)
                        .load(getDrawableForCategoryName(resultLower))
                        .into(resultsImage);

                if (resultLower.contains("fly") || resultLower.contains("destroy")) {
                    //resultsCard.setCardBackgroundColor(ContextCompat.getColor(ClassifierActivity.this, R.color.poisonousBackground));
                } else {
                    //resultsCard.setCardBackgroundColor(ContextCompat.getColor(ClassifierActivity.this, R.color.edibleBackground));
                }

                // fade in
                if (resultsCard.getVisibility() == View.GONE){
                    resultsCard.setAlpha(0f);
                    resultsCard.setScaleX(0.5f);
                    resultsCard.setScaleY(0.5f);
                    resultsCard.setVisibility(View.VISIBLE);
                    resultsCard.animate()
                            .alpha(1f)
                            .scaleX(1f)
                            .scaleY(1f)
                            .setDuration(500)
                            .setListener(null);
                }

            } else {
                // fade out
                if (resultsCard.getVisibility() == View.VISIBLE){
                    resultsCard.animate()
                            .alpha(0f)
                            .scaleX(0.5f)
                            .scaleY(0.5f)
                            .setDuration(500)
                            .setListener(new AnimatorListenerAdapter() {
                                @Override
                                public void onAnimationEnd(Animator animation) {
                                    resultsCard.setVisibility(View.GONE);
                                }
                            });
                }
            }


        });
    }


    @DrawableRes
    private int getDrawableForCategoryName(String name) {
        if (name.contains("morchella")) {
            return R.drawable.morchella_esculenta;
        } else if (name.contains("muscaria")) {
            return R.drawable.amanita_muscaria;
        } else if (name.contains("virosa")) {
            return R.drawable.amanita_virosa;
        } else if (name.contains("grifola")) {
            return R.drawable.grifola_frondosa;
        } else if (name.contains("edulis")) {
            return R.drawable.boletus_edulis;
        } else if (name.contains("coprinus")) {
            return R.drawable.coprinus_comatus;
        } else if (name.contains("cantharellus")) {
            return R.drawable.cantharellus_cibarius;
        } else if (name.contains("armillaria")) {
            return R.drawable.armillaria_mellea;
        } else if (name.contains("agaricus")) {
            return R.drawable.agaricus_bisporus;
        }
        return 0;
    }

    private String getCommonNameForBinomial(String name) {
        if (name.contains("morchella")) {
            return "Morel";
        } else if (name.contains("muscaria")) {
            return "Fly agaric";
        } else if (name.contains("grifola")) {
            return "Hen of the woods";
        } else if (name.contains("virosa")) {
            return "Destroying angel";
        } else if (name.contains("agaricus")) {
            return "Button mushroom";
        } else if (name.contains("cantharellus")) {
            return "Chanterelle";
        } else if (name.contains("coprinus")) {
            return "Shaggy mane";
        } else if (name.contains("edulis")) {
            return "Porcino";
        } else if (name.contains("armillaria")) {
            return "Honey mushroom";
        }
        return "";
    }
}
