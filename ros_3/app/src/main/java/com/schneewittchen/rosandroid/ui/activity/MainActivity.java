package com.schneewittchen.rosandroid.ui.activity;

import static java.lang.Thread.sleep;

import android.annotation.SuppressLint;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.schneewittchen.rosandroid.R;
import com.schneewittchen.rosandroid.model.entities.widgets.BaseEntity;
import com.schneewittchen.rosandroid.ui.general.WidgetEditListener;
import com.schneewittchen.rosandroid.widgets.joystick.JoystickData;
import com.schneewittchen.rosandroid.widgets.joystick.JoystickView;

import org.jboss.netty.buffer.ChannelBuffer;
import org.ros.address.InetAddressFactory;
import org.ros.android.BitmapFromCompressedImage;
import org.ros.android.MessageCallable;
import org.ros.android.view.RosImageView;
import org.ros.message.MessageListener;
import org.ros.namespace.GraphName;
import org.ros.node.ConnectedNode;
import org.ros.node.DefaultNodeMainExecutor;
import org.ros.node.Node;
import org.ros.node.NodeConfiguration;
import org.ros.node.NodeMain;
import org.ros.node.NodeMainExecutor;
import org.ros.node.topic.Publisher;
import org.ros.node.topic.Subscriber;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import sensor_msgs.CompressedImage;
import std_msgs.Int32;
import std_msgs.Int32MultiArray;
import std_msgs.String;
import std_msgs.UInt32;
import std_msgs.UInt32MultiArray;

import org.ros.android.RosActivity;

import java.util.Timer;
import android.os.Bundle;
import android.widget.ImageView;



import java.net.URI;
import java.util.TimerTask;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import java.util.stream.IntStream;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;


public class MainActivity extends  RosActivity implements NodeMain  {
    public MainActivity() {
        super("ROS String Publisher", "ROS String Publisher");
    }

    private Publisher<UInt32MultiArray> publisher_1;
    //private Publisher<UInt32> publisher_1;
    private Timer timer;
    private WebView webView;


    ///*
    java.lang.String URL_RGB = "http://192.168.0.141:8080/RGB_video_feed";
    java.lang.String URL_Fire = "http://192.168.0.141:8000/Fire_video_feed";
    java.lang.String URL_Slam = "http://192.168.0.141:8080/Slam_feed";
    java.lang.String Master_URI = "http://192.168.0.141:11311";

     //*/

    /*
    java.lang.String URL_RGB = "http://192.168.70.200:8080/RGB_video_feed";
    java.lang.String URL_Fire = "http://192.168.70.200:8000/Fire_video_feed";
    java.lang.String URL_Slam = "http://192.168.70.200:8000/SLam_feed";
    java.lang.String Master_URI = "http://192.168.70.200:11311";
    */

    SeekBar seekBarR;
    SeekBar seekBarL;
    TextView textView1;
    TextView textView2;
    TextView textView3;
    Button btnLDown;
    Button btnLUp;
    Button btnRDown;
    Button btnRUp;
    ToggleButton infoBtn;
    Switch camSw;
    Switch motorSw;
    TextView textView5;
    ImageButton emergencyButton;

    int ros = 50;
    int[] rosdata = {50, 50, 1, 1, 0, 0, 1, 1}; //seekbarR[0], seekbarL[1], buttonL[2], buttonR[3], 모터 동기화[4], 긴급정지[5], xPos[6], yPos[7]
    int camerastate = 0;
    float xPos = 0;
    float yPos = 0;
    float pre_xPos = 0;
    float pre_yPos = 0;

    @SuppressLint("WrongViewCast")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // ROS Sub
        NodeMainExecutor nodeMainExecutor = DefaultNodeMainExecutor.newDefault();
        NodeConfiguration nodeConfiguration =
                NodeConfiguration.newPublic(InetAddressFactory.newNonLoopback().getHostAddress(),
                        getMasterUri());
        nodeMainExecutor.execute(this, nodeConfiguration);


        seekBarL = findViewById(R.id.seekBar_L);
        seekBarR = findViewById(R.id.seekBar_R);
        textView1 = findViewById(R.id.textView);
        textView2 = findViewById(R.id.textView2);
        textView3 = findViewById(R.id.textView3);
        btnLDown = findViewById(R.id.button);
        btnLUp = findViewById(R.id.button2);
        btnRDown = findViewById(R.id.button3);
        btnRUp = findViewById(R.id.button4);
        infoBtn = findViewById(R.id.toggleButton);
        camSw = findViewById(R.id.switch1);
        motorSw = findViewById(R.id.switch2);
        emergencyButton = findViewById(R.id.imageButton);

        webView = (WebView) findViewById(R.id.webview); // 레이아웃에서 WebView 찾기
        webView.setWebViewClient(new WebViewClient()); // WebView에 웹뷰 클라이언트 설정
        webView.getSettings().setJavaScriptEnabled(true); // 자바스크립트 허용
        webView.loadUrl(URL_RGB);

    }


    @Override
    public GraphName getDefaultNodeName() {
        return GraphName.of("ros_android");
    }


    @SuppressLint("ClickableViewAccessibility")
    @Override
    public void onStart(ConnectedNode connectedNode) {
        /*

        Subscriber<String> infoSubscriber1 = connectedNode.newSubscriber("/info1", String._TYPE);
        infoSubscriber1.addMessageListener((message) -> {
            runOnUiThread(() -> {
                textView5.setText("Info 1: " + message.getData());
            });
        });

        Subscriber<String> infoSubscriber2 = connectedNode.newSubscriber("/info2", String._TYPE);
        infoSubscriber2.addMessageListener((message) -> {
            runOnUiThread(() -> {
                textView2.setText("Info 2: " + message.getData());
            });
        });

         */



        publisher_1 = connectedNode.newPublisher("/motordata", UInt32MultiArray._TYPE);
        //publisher_1 = connectedNode.newPublisher("/motordata", UInt32._TYPE);

        timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                UInt32MultiArray message = publisher_1.newMessage();
                message.setData(rosdata);
                //UInt32 message = publisher_1.newMessage();
                //message.setData(ros);
                publisher_1.publish(message);
            }
        }, 10, 10);



        ////////////////////////////////////////////////////////////SEEKBARL
        seekBarL.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                rosdata[0] = i;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                seekBarL.setProgress(50, true);
                rosdata[0] = 50;
            }
        });

        ////////////////////////////////////////////////////////////SEEKBARR
        seekBarR.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                rosdata[1] = i;
                ros = i;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                seekBarR.setProgress(50, true);
                rosdata[1] = 50;
            }
        });

        ////////////////////////////////////////////////////////////BUTTONL
        btnLUp.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {

                int status = motionEvent.getAction();

                if(status == MotionEvent.ACTION_DOWN){
                    rosdata[2] = 2;
                }
                else if(status == MotionEvent.ACTION_UP){
                    rosdata[2] = 1;
                }

                return false;
            }
        });

        btnLDown.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {

                int status = motionEvent.getAction();

                if(status == MotionEvent.ACTION_DOWN){
                    rosdata[2] = 0;
                }
                else if(status == MotionEvent.ACTION_UP){
                    rosdata[2] = 1;
                }

                return false;
            }
        });

        ////////////////////////////////////////////////////////////BUTTONR
        btnRUp.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {

                int status = motionEvent.getAction();

                if(status == MotionEvent.ACTION_DOWN){
                    rosdata[3] = 2;
                }
                else if(status == MotionEvent.ACTION_UP){
                    rosdata[3] = 1;
                }

                return false;
            }
        });

        btnRDown.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {

                int status = motionEvent.getAction();

                if(status == MotionEvent.ACTION_DOWN){
                    rosdata[3] = 0;
                }
                else if(status == MotionEvent.ACTION_UP){
                    rosdata[3] = 1;
                }

                return false;
            }
        });






        infoBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked) {
                    webView.loadUrl(URL_Slam);
                    /*
                    textView1.setVisibility(View.VISIBLE);
                    textView2.setVisibility(View.VISIBLE);
                    textView3.setVisibility(View.VISIBLE);

                     */
                } else {
                    if (camerastate == 1){
                        webView.loadUrl(URL_Fire);
                    }
                    else{
                        webView.loadUrl(URL_RGB);
                    }
                    /*
                    textView1.setVisibility(View.INVISIBLE);
                    textView2.setVisibility(View.INVISIBLE);
                    textView3.setVisibility(View.INVISIBLE);

                     */

                }
            }
        });

        camSw.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked) {
                    camerastate = 1;
                    camSw.setText("열화상 모드");
                    webView.loadUrl(URL_Fire);
                } else {
                    camerastate = 0;
                    camSw.setText("RGB 모드");
                    webView.loadUrl(URL_RGB);
                }
            }
        });

        motorSw.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked) {
                    rosdata[4] = 1;
                    btnLUp.setVisibility(View.INVISIBLE);
                    btnLDown.setVisibility(View.INVISIBLE);
                } else {
                    rosdata[4] = 0;
                    btnLUp.setVisibility(View.VISIBLE);
                    btnLDown.setVisibility(View.VISIBLE);
                }
            }
        });

        emergencyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                rosdata[5] = 1;
                Toast.makeText(getApplicationContext(), "긴급정지합니다!!!", Toast.LENGTH_LONG).show();
            }
        });

        webView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {

                int status = motionEvent.getAction();


                if (status == motionEvent.ACTION_DOWN){
                    pre_xPos = motionEvent.getX();
                    pre_yPos = motionEvent.getY();
                }
                else if(status == MotionEvent.ACTION_MOVE){
                    xPos = motionEvent.getX();
                    yPos = motionEvent.getY();

                    int moveX = (int) (xPos - pre_xPos);
                    int moveY = (int) (yPos - pre_yPos);


                    if (moveX > 150) {
                        rosdata[6] = 2; //right
                    }
                    else if (moveX <= -150) {
                        rosdata[6] = 0; //left
                    }
                    else {
                        rosdata[6] = 1;
                    }

                    if (moveY > 100) {
                        rosdata[7] = 0; //down
                    }
                    else if (moveY <= -100) {
                        rosdata[7] = 2; //up
                    }
                    else {
                        rosdata[7] = 1;
                    }

                }
                else if(status == MotionEvent.ACTION_UP){
                    rosdata[6] = 1;
                    rosdata[7] = 1;
                }

                Log.d("move", "x:" + rosdata[6] + ",  y:" + rosdata[7]);

                return false;


            }

        });



    }



    @Override
    public void onShutdown(Node node) {
    }

    @Override
    public void onShutdownComplete(Node node) {
    }

    @Override
    public void onError(Node node, Throwable throwable) {
    }

    public URI getMasterUri() {
        // Replace this with the URI of the ROS master
        return URI.create(Master_URI);
    }

    @Override
    protected void init(NodeMainExecutor nodeMainExecutor) {

    }

    // NodeMainExecutorService 클래스를 생성합니다.
    private class NodeMainExecutorService extends org.ros.android.NodeMainExecutorService {

    }

}
