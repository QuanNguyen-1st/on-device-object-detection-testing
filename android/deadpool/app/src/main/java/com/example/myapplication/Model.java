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
import android.os.Environment;
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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.time.Clock;
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
    private int numChannel = 0; // CLASSES + 1 + 4
    private int numBoxes = 0; // BOXES


    // Static anchor boxes of model
    private static final int NUM_FEATURES = 6;
    private static final int[] NUM_BOXES = new int[]{4,6,6,6,4,4};
    private static final int[] LAYER_WIDTHS = new int[]{28, 14, 7, 4, 2, 1};
    private static final float[] ASP = new float[]{0.333f,0.5f,1.0f,1.25f,1.5f,2};
    private static final float MIN_SCALE = 0.2f;
    private static final float MAX_SCALE = 0.9f;

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
    private static final float CONFIDENCE_THRESHOLD = 0.5F;
    private static final float IOU_THRESHOLD = 0.1F;
    private ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(new CastOp(INPUT_IMAGE_TYPE))
            .build();

    // Train info3
    private static final int NUM_EPOCHS = 25;
    private static final int BATCH_SIZE = 1;
    private static final int NUM_TRAINING = 39;
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
        restore();

        int[] inputShape = interpreter.getInputTensor(0).shape();
        int[] outputShape = interpreter.getOutputTensor(0).shape();

        tensorWidth = inputShape[1];
        tensorHeight = inputShape[2];
//        Log.d("tensorSize", tensorWidth + ", " + tensorHeight);
        numBoxes = outputShape[1];
        numChannel = outputShape[2];
//        Log.d("tensorSize", numBoxes + ", " + numChannel);

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
        float scale_step = (MAX_SCALE - MIN_SCALE) / (NUM_FEATURES - 1);
        for (int i = 0; i < NUM_FEATURES; i++) {
            size += LAYER_WIDTHS[i] * LAYER_WIDTHS[i] * NUM_BOXES[i];
            scales[i] = MIN_SCALE + scale_step * i;
        }

        assert(size == numBoxes);

        for (int i = 0; i < ASP.length; i++) {
            asp1[i] = (float) Math.sqrt(ASP[i]);
            asp2[i] = 1 / asp1[i];
        }
        anchorBoxesCenters = new float[size * 2];
        anchorBoxesWHs = new float[size * 2];
        int idx = 0;
        for (int i = 0; i < NUM_FEATURES; i++) {
            int gridSize = LAYER_WIDTHS[i];
            int numBox = NUM_BOXES[i];
            float scale = scales[i];

            for (int j = 0; j < gridSize; j++) {
                for (int k = 0; k < gridSize; k++) {
                    float cx = (float)(k + 0.5) / gridSize;
                    float cy = (float)(j + 0.5) / gridSize;

                    for (int l = 0; l < numBox; l++) {
                        float anchor_width = scale * asp1[l];
                        float anchor_height = scale * asp2[l];

                        anchorBoxesCenters[idx*2] = cx;
                        anchorBoxesCenters[idx*2 + 1] = cy;

                        anchorBoxesWHs[idx*2] = anchor_width;
                        anchorBoxesWHs[idx*2 + 1] = anchor_height;

                        idx += 1;
                    }
                }
            }
        }
    }

    public Bitmap drawBoundingBoxes(Bitmap bitmap, List<BoundingBox> boxes) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5f);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40f);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);

        for (BoundingBox box : boxes) {
            RectF rect = new RectF(
                    box.x1 * mutableBitmap.getWidth(),
                    box.y1 * mutableBitmap.getHeight(),
                    box.x2 * mutableBitmap.getWidth(),
                    box.y2 * mutableBitmap.getHeight()
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
                float cx = (anchorBoxesCenters[box_idx] + delta_cx / tensorWidth);
                float cy = (anchorBoxesCenters[box_idx + 1] + delta_cy / tensorHeight);
                float w = (anchorBoxesWHs[box_idx] + delta_w / tensorWidth);
                float h = (anchorBoxesWHs[box_idx + 1] + delta_h / tensorHeight);
//                Log.d("AAAAA", maxConf + " " + cx + " " + cy + " " + w + " " + h);
                float x1 = cx - (w / 2F);
                float y1 = cy - (h / 2F);
                float x2 = cx + (w / 2F);
                float y2 = cy + (h / 2F);

                if (x1 < 0F) x1 = 0;
                if (x1 > 1F) x1 = 1;
                if (y1 < 0F) y1 = 0;
                if (y1 > 1F) y1 = 1;
                if (x2 < 0F) x2 = 0;
                if (x2 > 1F) x2 = 1;
                if (y2 < 0F) y2 = 0;
                if (y2 > 1F) y2 = 1;

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
//        return boundingBoxes;
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

    private float[] convertY(float[] y, int z) {
        List<Integer> boxIdx = new ArrayList<>();
        float[] output = new float[numBoxes * 5];
        for (int i = 0; i < numBoxes; i++) {
            output[5*i] = numChannel - 5;
        }
        for (int i = 0; i < y.length / 5; i++) {
            int cls = (int) y[5*i];
            float x1 = y[5*i + 1];
            float y1 = y[5*i + 2];
            float x2 = y[5*i + 3];
            float y2 = y[5*i + 4];

            float cx = (x1 + x2) / 2;
            float cy = (y1 + y2) / 2;
            float w = x2 - x1;
            float h = y2 - y1;

            BoundingBox bbox = new BoundingBox(
                    x1, y1, x2, y2,
                    cx, cy, w, h,
                    1, cls
            );
            for (int j = 0; j < numBoxes; j++) {
                float anchor_cx = anchorBoxesCenters[2*j];
                float anchor_cy = anchorBoxesCenters[2*j + 1];
                float anchor_w = anchorBoxesWHs[2*j];
                float anchor_h = anchorBoxesWHs[2*j + 1];

                float anchor_x1 = anchor_cx - (anchor_w / 2F);
                float anchor_y1 = anchor_cy - (anchor_h / 2F);
                float anchor_x2 = anchor_cx + (anchor_w / 2F);
                float anchor_y2 = anchor_cy + (anchor_h / 2F);

                if (calculateIoU(bbox, new BoundingBox(
                        anchor_x1, anchor_y1, anchor_x2, anchor_y2,
                        anchor_cx, anchor_cy, anchor_w, anchor_h,
                        1, numChannel - 5)) >= 0.5f){
                    boxIdx.add(j);
                    float delta_cx = (cx - anchor_cx) * tensorWidth;
                    float delta_cy = (cy - anchor_cy) * tensorHeight;
                    float delta_w = (w - anchor_w) * tensorWidth;
                    float delta_h = (h - anchor_h) * tensorHeight;

                    output[5*j] = cls;
                    output[5*j + 1] = delta_cx;
                    output[5*j + 2] = delta_cy;
                    output[5*j + 3] = delta_w;
                    output[5*j + 4] = delta_h;
                }
            }
        }
        int[] array = new int[boxIdx.size()];
        for (int i = 0; i < boxIdx.size(); i++) {
            array[i] = boxIdx.get(i);
        }
//        WriteToDownload(array, z);
        return output;
    }

    private ByteBuffer readTestTrainImages(int idx) {
        AssetManager assetManager = context.getAssets();
        ByteBuffer buffer = ByteBuffer.allocate(tensorWidth * tensorHeight * 3 * 4 * BATCH_SIZE).order(ByteOrder.nativeOrder());
        for (int i = 0; i < BATCH_SIZE; i++) {
            int fileIdx = idx * BATCH_SIZE + i + 1;
            String imageName = "deadpool/images/" + fileIdx + ".jpg";
            try (InputStream inputStream = assetManager.open(imageName)) {
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false);
                TensorImage tensorImage = new TensorImage(INPUT_IMAGE_TYPE);
                tensorImage.load(resizedBitmap);
                TensorImage processedImage = imageProcessor.process(tensorImage);

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
            int fileIdx = idx * BATCH_SIZE + i + 1;
            String labelName = "deadpool/labels/" + fileIdx + ".txt";
            try (InputStream inputStream = assetManager.open(labelName)) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                List<Float> floatList = new ArrayList<>();
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split("\\s+");
                    for (String part : parts) {
                        floatList.add(Float.parseFloat(part));
                    }

                }
                float[] y = new float[floatList.size()];
                for (int j = 0; j < floatList.size(); j++) {
                    y[j] = floatList.get(j);
                }
                float[] output = convertY(y, fileIdx);
                for (float v : output) {
                    buffer.putFloat(v);
                }

            }
            catch (IOException e) {
                Log.e("Model", "Error loading label: " + labelName, e);
            };
        }
        buffer.rewind();
        return buffer;
    }

    public void train(int num_train) {
        float start = System.nanoTime();
        int num_batches = num_train / BATCH_SIZE ;
        List<ByteBuffer> trainImageBatches = new ArrayList<>(num_batches);
        List<ByteBuffer> trainLabelBatches = new ArrayList<>(num_batches);

        for (int i = 0; i < num_batches; i++) {
            ByteBuffer imageBuffer = ByteBuffer.allocate(tensorWidth * tensorHeight * 3 * 4 * BATCH_SIZE).order(ByteOrder.nativeOrder());
            ByteBuffer labelBuffer = ByteBuffer.allocate(numBoxes * 5 * 4 * BATCH_SIZE).order(ByteOrder.nativeOrder());

            //TODO: Insert images and labels data as byte buffer in batches to train


            imageBuffer.position(0);
            labelBuffer.position(0);

            trainImageBatches.add(imageBuffer);
            trainLabelBatches.add(labelBuffer);
        }

        Log.d("Model Training", "Load data done in: " + ((System.nanoTime() - start) / 10E9) + " ms");

        for (int i = 0; i < NUM_EPOCHS; i++) {
            start = System.nanoTime();
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

            Log.d("Model Training", "Epoch: " + (i + 1) + ", Loss: " + lossTensorBuffer.getFloatArray()[0] + ", Time: " + ((System.nanoTime() - start) / 10E9) + " ms");
        }
        save();
    }

    public void testTrain() {
        List<ByteBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
        List<ByteBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);

        for (int i = 0; i < NUM_BATCHES; i++) {
            ByteBuffer imageBuffer = readTestTrainImages(i);
            ByteBuffer labelBuffer = readTestTrainLabels(i);

            imageBuffer.position(0);
            labelBuffer.position(0);

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
        save();
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

    private void WriteToDownload(int[] array, int idx) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int val : array) {
            stringBuilder.append(val).append("\n");
        }
        String intString = stringBuilder.toString();
        FileOutputStream fileOutputStream = null;
        try {
            // Create or open the file in the app's internal storage
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), idx + ".txt");
            fileOutputStream = new FileOutputStream(file);
            fileOutputStream.write(intString.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void WriteToDownload(ByteBuffer byteBuffer, int idx) {
        byteBuffer.order(ByteOrder.nativeOrder());
        int numFloats = byteBuffer.remaining() / Float.BYTES;
        float[] floatArray = new float[numFloats];
        byteBuffer.asFloatBuffer().get(floatArray);

        StringBuilder stringBuilder = new StringBuilder();
        for (float val : floatArray) {
            stringBuilder.append(val).append("\n");
        }
        String floatString = stringBuilder.toString();

        FileOutputStream fileOutputStream = null;
        try {
            // Create or open the file in the app's internal storage
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), idx + ".txt");
            fileOutputStream = new FileOutputStream(file);
            fileOutputStream.write(floatString.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
