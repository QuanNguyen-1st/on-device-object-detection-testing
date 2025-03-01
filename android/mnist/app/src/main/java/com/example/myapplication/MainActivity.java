package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private Model model;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ImageView imageView = findViewById(R.id.imageView);
        try {
            InputStream inputStream = getAssets().open("mnist/images/299.png");
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            model = new Model(this);
            model.testTrain();
            imageView.setImageBitmap(model.predict(bitmap));
        }
        catch (IOException e) {

        }

    }
}