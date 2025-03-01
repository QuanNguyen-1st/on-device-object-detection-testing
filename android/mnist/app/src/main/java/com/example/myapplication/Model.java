package com.example.myapplication;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Model {
    private Context context;

    // Model and interpreter info
    private static final String MODEL_PATH = "train.tflite";
    private Interpreter interpreter = null;
    private int tensorWidth = 0;
    private int tensorHeight = 0;
    private int numChannel = 0; // 15
    private int numBoxes = 0; // BOXES


    // Static anchor boxes of model
    private static final int NUM_FEATURES = 3;
    private static final int[] NUM_BOXES = new int[]{3,3,3};
    private static final int[] LAYER_WIDTHS = new int[]{28, 14, 7};
    private static final float[] ASP = new float[]{0.5f, 1, 1.5f};
    private static final float MIN_SCALE = 0.1f;
    private static final float MAX_SCALE = 1.5f;

    private static final float[] asp1 = new float[ASP.length];
    private static final float[] asp2 = new float[ASP.length];
    private static final float[] scales = new float[NUM_FEATURES];

    private static float[] anchorBoxesCenters;
    private static float[] anchorBoxesWHs;


    // Data input and output process
    private static final float INPUT_MEAN = 127.5f;
    private static final float INPUT_STANDARD_DEVIATION = 127.5f;
    private static final DataType INPUT_IMAGE_TYPE = DataType.FLOAT32;
    private static final DataType OUTPUT_IMAGE_TYPE = DataType.FLOAT32;
    private static final float CONFIDENCE_THRESHOLD = 0.3F;
    private static final float IOU_THRESHOLD = 0.5F;
    private ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(new CastOp(INPUT_IMAGE_TYPE))
            .build();

    // Train info
    private static final int NUM_EPOCHS = 10;
    private static final int BATCH_SIZE = 1;
    private static final int NUM_TRAINING = 40;
    private static final int NUM_BATCHES = NUM_TRAINING / BATCH_SIZE;

    public class BoundingBox {
        float x1, y1, x2, y2, cx, cy, w, h, cnf;
        int cls;

        public BoundingBox(float x1, float y1, float x2, float y2,
                           float cx, float cy, float w, float h,
                           float cnf, int cls) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.cx = cx;
            this.cy = cy;
            this.w = w;
            this.h = h;
            this.cnf = cnf;
            this.cls = cls;
        }
    }

    public Model(Context context) throws IOException {
        this.context = context;
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(FileUtil.loadMappedFile(context, MODEL_PATH), options);
//        restore();

        int[] inputShape = interpreter.getInputTensor(0).shape();
        int[] outputShape = interpreter.getOutputTensor(0).shape();

        tensorWidth = inputShape[1];
        tensorHeight = inputShape[2];
        numBoxes = outputShape[1];
        numChannel = outputShape[2];

        generateAnchorBoxes();
    }

    private float[] softmax(float[] logits) {
        float[] expScores = new float[logits.length];
        float sum = 0f;

        for (int i = 0; i < logits.length; i++) {
            expScores[i] = (float) Math.exp(logits[i]);
            sum += expScores[i];
        }

        for (int i = 0; i < expScores.length; i++) {
            expScores[i] /= sum;
        }

        return expScores;
    }

    private void generateAnchorBoxes() {
        int size = 0;
        for (int i = 0; i < NUM_FEATURES; i++) {
            size += LAYER_WIDTHS[i] * LAYER_WIDTHS[i] * NUM_BOXES[i];
            scales[i] = (MAX_SCALE - MIN_SCALE) * (NUM_FEATURES - i - 1) / NUM_FEATURES + MIN_SCALE;
        }

        assert(size == numBoxes);

        for (int i = 0; i < ASP.length; i++) {
            asp1[i] = (float) Math.sqrt(ASP[i]);
            asp2[i] = 1 / asp1[i];
        }
        anchorBoxesCenters = new float[size * 2];
        anchorBoxesWHs = new float[size * 2];
        int idx = 0;
        for (int l = 0; l < NUM_FEATURES; l++) {
            int gridSize = LAYER_WIDTHS[l];
            int numBox = NUM_BOXES[l];
            float scale = scales[l];
            float step_size = tensorWidth * 1f / gridSize;

            for (int i = 0; i < gridSize; i++) {
                for (int j = 0; j < gridSize; j++) {
                    int pos = idx + (i * gridSize + j) * numBox;

                    for (int k = 0; k < numBox; k++) {
                        anchorBoxesCenters[(pos + k) * 2] = i * step_size + step_size / 2;
                        anchorBoxesCenters[(pos + k) * 2 + 1] = j * step_size + step_size / 2;
                    }

                    for (int k = 0; k < numBox; k++) {
                        anchorBoxesWHs[(pos + k) * 2] = gridSize * scale * asp1[k]; // width
                        anchorBoxesWHs[(pos + k) * 2 + 1] = gridSize * scale * asp2[k]; // height
                    }
                }
            }
            idx += gridSize * gridSize * numBox;
        }
    }

    public Bitmap drawBoundingBoxes(Bitmap bitmap, List<BoundingBox> boxes) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1f);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40f);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);

        for (BoundingBox box : boxes) {
            RectF rect = new RectF(
                    box.y1 * mutableBitmap.getWidth(),
                    box.x1 * mutableBitmap.getHeight(),
                    box.y2 * mutableBitmap.getWidth(),
                    box.x2 * mutableBitmap.getHeight()
            );
            canvas.drawRect(rect, paint);
        }

        return mutableBitmap;
    }

    public List<BoundingBox> bestBox(float[] array) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();

        for (int b = 0; b < numBoxes; b++) {
            int offset = b * numChannel;
            float delta_cx = array[offset + numChannel - 4];
            float delta_cy = array[offset + numChannel - 3];
            float delta_w = array[offset + numChannel - 2];
            float delta_h = array[offset + numChannel - 1];

            float[] conf = new float[numChannel - 4];
            for (int i = 0; i < numChannel - 4; i++) {
                conf[i] = array[offset + i];
            }
            conf = softmax(conf);
            float maxConf = conf[0];
            int maxIdx = 0;
            for (int i = 1; i < numChannel - 4; i++) {
                if (conf[i] > maxConf) {
                    maxConf = conf[i];
                    maxIdx = i;
                }
            }
            if (maxConf > CONFIDENCE_THRESHOLD && maxIdx != numChannel - 4 - 1) {
                int box_idx = b * 2;
                float cx = (anchorBoxesCenters[box_idx] + delta_cx) / tensorWidth;
                float cy = (anchorBoxesCenters[box_idx + 1] + delta_cy) / tensorHeight;
                float w = (anchorBoxesWHs[box_idx] + delta_w) / tensorWidth;
                float h = (anchorBoxesWHs[box_idx + 1] + delta_h) / tensorHeight;
//                Log.d("AAAAA", maxConf + " " + cx + " " + cy + " " + w + " " + h);
                float x1 = cx - (w / 2F);
                float y1 = cy - (h / 2F);
                float x2 = cx + (w / 2F);
                float y2 = cy + (h / 2F);
                if (x1 < 0F || x1 > 1F) continue;
                if (y1 < 0F || y1 > 1F) continue;
                if (x2 < 0F || x2 > 1F) continue;
                if (y2 < 0F || y2 > 1F) continue;

                boundingBoxes.add(
                        new BoundingBox(
                                x1, y1, x2, y2,
                                cx, cy, w, h,
                                maxConf, maxIdx
                        )
                );
            }
        }

        if (boundingBoxes.isEmpty()) return null;
        return applyNMS(boundingBoxes);
    }

    private List<BoundingBox> applyNMS(List<BoundingBox> boxes) {
        boxes.sort((o1, o2) -> Float.compare(o2.cnf, o1.cnf));
        List<BoundingBox> selectedBoxes = new ArrayList<>();

        while (!boxes.isEmpty()) {
            BoundingBox first = boxes.get(0);
            selectedBoxes.add(first);
            boxes.remove(0);

            boxes.removeIf(nextBox -> calculateIoU(first, nextBox) >= IOU_THRESHOLD);
        }

        return selectedBoxes;
    }

    private float calculateIoU(BoundingBox box1, BoundingBox box2) {
        float x1 = Math.max(box1.x1, box2.x1);
        float y1 = Math.max(box1.y1, box2.y1);
        float x2 = Math.min(box1.x2, box2.x2);
        float y2 = Math.min(box1.y2, box2.y2);
        float intersectionArea = Math.max(0F, x2 - x1) * Math.max(0F, y2 - y1);
        float box1Area = box1.w * box1.h;
        float box2Area = box2.w * box2.h;
        return intersectionArea / (box1Area + box2Area - intersectionArea);
    }

    public Bitmap predict(Bitmap bitmap) {
        float start = System.currentTimeMillis();
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false);
        TensorImage tensorImage = new TensorImage(INPUT_IMAGE_TYPE);
        tensorImage.load(resizedBitmap);
        TensorImage processedImage = imageProcessor.process(tensorImage);
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, numBoxes, numChannel}, OUTPUT_IMAGE_TYPE);

        interpreter.run(processedImage.getBuffer(), outputBuffer.getBuffer());

        float[] outputArray = outputBuffer.getFloatArray();
        List<BoundingBox> bestBoxes = bestBox(outputArray);
        Log.d("Time Infer", (System.currentTimeMillis() - start) + "");
        if (bestBoxes != null) {
            return drawBoundingBoxes(bitmap, bestBoxes);
        }
        return bitmap;
    }

    private ByteBuffer readTestTrainImages(int idx) {
        AssetManager assetManager = context.getAssets();
        ByteBuffer buffer = ByteBuffer.allocate(tensorWidth * tensorHeight * 3 * 4 * BATCH_SIZE).order(ByteOrder.nativeOrder());
        for (int i = 0; i < BATCH_SIZE; i++) {
            int fileIdx = idx * BATCH_SIZE + i;
            String imageName = "mnist/images/" + fileIdx + ".png";
            try (InputStream inputStream = assetManager.open(imageName)) {
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false);
                TensorImage tensorImage = new TensorImage(INPUT_IMAGE_TYPE);
                tensorImage.load(resizedBitmap);
                TensorImage processedImage = imageProcessor.process(tensorImage);

//                ByteBuffer byteBuffer = processedImage.getBuffer();
//                byteBuffer.order(ByteOrder.nativeOrder());
//                int numFloats = byteBuffer.remaining() / Float.BYTES;
//                float[] floatArray = new float[numFloats];
//                byteBuffer.asFloatBuffer().get(floatArray);

                buffer.put(processedImage.getBuffer());
            }
            catch (IOException e) {
                Log.e("Model", "Error loading image: " + imageName, e);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private ByteBuffer readTestTrainLabels(int idx) {
        AssetManager assetManager = context.getAssets();
        ByteBuffer buffer = ByteBuffer.allocate(numBoxes * 5 * 4 * BATCH_SIZE).order(ByteOrder.nativeOrder());
        for (int i = 0; i < BATCH_SIZE; i++) {
            int fileIdx = idx * BATCH_SIZE + i;
            String labelName = "mnist/labels/" + fileIdx + ".txt";
            try (InputStream inputStream = assetManager.open(labelName)) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    for (String part : parts) {
                        buffer.putFloat(Float.parseFloat(part));
                    }
                }
            } catch (IOException e) {
                Log.e("Model", "Error loading label: " + labelName, e);
            };
        }
        buffer.rewind();
//        ByteBuffer byteBuffer = buffer;
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int numFloats = byteBuffer.remaining() / Float.BYTES;
//        float[] floatArray = new float[numFloats];
//        byteBuffer.asFloatBuffer().get(floatArray);
//        Log.d("DDDDD", Arrays.toString(floatArray));
        return buffer;
    }

    public void testTrain() {
//        Interpreter.Options options = new Interpreter.Options();
//        options.setNumThreads(4);
//        Interpreter anotherInterpreter = null;
//        try {
//            anotherInterpreter = new Interpreter(FileUtil.loadMappedFile(context, "train.tflite"), options);
//        }
//        catch (IOException e) {
//
//        }

        List<ByteBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
        List<ByteBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);

        for (int i = 0; i < NUM_BATCHES; i++) {
            ByteBuffer imageBuffer = readTestTrainImages(i);
            ByteBuffer labelBuffer = readTestTrainLabels(i);

//            imageBuffer.position(0);
//            labelBuffer.position(0);

            trainImageBatches.add(imageBuffer);
            trainLabelBatches.add(labelBuffer);
        }

        for (int i = 0; i < NUM_EPOCHS; i++) {
            Map<String, Object> outputs = new HashMap<>();
            FloatBuffer loss = FloatBuffer.allocate(1);
//            outputs.put("loss", loss);

            for (int batchIdx = 0; batchIdx< NUM_BATCHES; batchIdx ++) {
                ByteBuffer imageBuffer = trainImageBatches.get(batchIdx);
                ByteBuffer labelBuffer = trainLabelBatches.get(batchIdx);

                TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{BATCH_SIZE, tensorWidth, tensorHeight, 3}, INPUT_IMAGE_TYPE);
                inputBuffer.loadBuffer(imageBuffer);

                TensorBuffer labelBufferTensor = TensorBuffer.createFixedSize(new int[]{BATCH_SIZE, numBoxes, 5}, OUTPUT_IMAGE_TYPE);
                labelBufferTensor.loadBuffer(labelBuffer);

                Map<String, Object> inputs = new HashMap<>();
                inputs.put("images", inputBuffer.getBuffer());
                inputs.put("gt", labelBufferTensor.getBuffer());

                interpreter.runSignature(inputs, outputs, "train");
            }
            // Retrieve the loss tensor directly from the interpreter
            TensorBuffer lossTensorBuffer = TensorBuffer.createFixedSize(new int[]{1}, DataType.FLOAT32);
            ByteBuffer lossByteBuffer = interpreter.getOutputTensorFromSignature("loss", "train").asReadOnlyBuffer();
            lossTensorBuffer.loadBuffer(lossByteBuffer);

            Log.d("Model Training", "Epoch: " + (i + 1) + ", Loss: " + lossTensorBuffer.getFloatArray()[0]);
        }
//        save();
    }

    public void save() {
        File outputFile = new File(context.getFilesDir(), "checkpoint.ckpt");
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", outputFile.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();
        interpreter.runSignature(inputs, outputs, "save");
    }

    public void restore() {
        File outputFile = new File(context.getFilesDir(), "checkpoint.ckpt");
        if (outputFile.exists()) {
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("checkpoint_path", outputFile.getAbsolutePath());
            Map<String, Object> outputs = new HashMap<>();
            interpreter.runSignature(inputs, outputs, "restore");
        }
    }
}
