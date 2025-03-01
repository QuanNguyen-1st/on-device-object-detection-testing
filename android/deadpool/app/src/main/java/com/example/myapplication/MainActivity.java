package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Random;

public class MainActivity extends AppCompatActivity {
    private Model model;
    private final int MAX_IDX = 39;
    private final int MIN_IDX = 1;
    private int curr_idx = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ImageView imageView = findViewById(R.id.imageView);
        try {
            model = new Model(this);
//            model.testTrain();
        }
        catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            String fullStackTrace = sw.toString();
            Log.e("FullStackTrace", fullStackTrace);
        }
        Bitmap[] inferredImages = new Bitmap[MAX_IDX];
        for (int i = 0; i <= MAX_IDX - MIN_IDX; i++) {
            int idx = i + MIN_IDX;
            try {
                InputStream inputStream = getAssets().open("deadpool/images/" + idx + ".jpg");
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                inferredImages[i] = model.predict(bitmap);
            }
            catch (IOException e) {
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                String fullStackTrace = sw.toString();
                Log.e("FullStackTrace", fullStackTrace);
            }
        }
        imageView.setImageBitmap(inferredImages[curr_idx + MIN_IDX]);
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                curr_idx += 1;
                if (curr_idx + MIN_IDX >= MAX_IDX) curr_idx = 0;
                imageView.setImageBitmap(inferredImages[curr_idx + MIN_IDX]);
            }

        });
    }
}