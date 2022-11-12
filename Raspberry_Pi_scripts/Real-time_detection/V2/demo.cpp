// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include "/home/pi/Desktop/src_files/include/MNN_UltraFace.hpp"
#include <chrono>
#include <pigpio.h>
#include <thread>

// set parameters before you run
const bool videoOutput = false;
const bool useNCNNorMNN = true;         // true for NCNN, false for MNN
const bool useSlim320Model = true;      // true for slim-320 face model, false for custom face detection model
const bool quantizationON_face = false;
const bool quantizationON_camera = true;
const float threshold_camera = 0.6;      // confidence threshold
const float threshold_face = 0.6;        // confidence threshold
const int numOfFramesToActivate = 5;     // set how many frames need to be detected, before glasses turn dark
const std::vector<float> anchors_camera {38.98,40.81, 77.58,94.43, 115.39,234.96, 179.64,115.53, 243.35,210.96, 284.08,313.04};
const std::vector<float> anchors_face {7.93,14.19, 18.07,31.59, 33.11,56.43, 57.30,90.38, 100.60,145.96, 200.42,230.04};

void wait(){
    std::cout << "Waiting for the button to be pressed..." << std::endl;
    gpioWrite(5, PI_LOW);
    gpioWrite(26, PI_LOW);
    while(gpioRead(19) == 1){
        continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    while(gpioRead(19) == 0){
        continue;
    }
}

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened()){
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    if (gpioInitialise() < 0) {
        return 1; // Failed to initialize.
    }

    gpioSetMode(19, PI_INPUT);
    wait();

    float f;
    float FPS[16];
    int i, Fcnt=0;
    std::chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    auto LED = std::chrono::steady_clock::now();

    gpioSetMode(5, PI_OUTPUT);
    gpioWrite(5, PI_LOW);
    gpioSetMode(26, PI_OUTPUT);
    gpioWrite(26, PI_LOW);

    int count = 0;

    // create yoloFastestv2 class instances
    yoloFastestv2NCNN api_cameraNCNN;
    yoloFastestv2MNN api_cameraMNN;
    yoloFastestv2NCNN api_faceNCNN;
    yoloFastestv2MNN api_faceMNN;
    UltraFace ultraface; // config model input

    if (useNCNNorMNN){
        api_cameraNCNN.init(1, anchors_camera);
        if (quantizationON_camera){
            api_cameraNCNN.loadModel("./model/yolo-fastestv2-int8_camera.param",
                      "./model/yolo-fastestv2-int8_camera.bin");
        }
        else{
            api_cameraNCNN.loadModel("./model/yolo-fastestv2-opt_camera.param",
                      "./model/yolo-fastestv2-opt_camera.bin");
        }
    }
    else{
        api_cameraMNN.init(1, anchors_camera);
        if (quantizationON_camera){
            api_cameraMNN.loadModel("./model/yolo-fastestv2-int8_camera.mnn");
        }
        else{
            api_cameraMNN.loadModel("./model/yolo-fastestv2-opt_camera.mnn");
        }
    }

    if (!useSlim320Model){
        if (useNCNNorMNN){
            api_faceNCNN.init(1, anchors_face);
            if (quantizationON_face){
                api_faceNCNN.loadModel("./model/yolo-fastestv2-int8_face.param",
                          "./model/yolo-fastestv2-int8_face.bin");
            }
            else{
                api_faceNCNN.loadModel("./model/yolo-fastestv2-opt_face.param",
                          "./model/yolo-fastestv2-opt_face.bin");
            }
        }
        else{
            api_faceMNN.init(1, anchors_face);
            if (quantizationON_face){
                api_faceMNN.loadModel("./model/yolo-fastestv2-int8_face.mnn");
            }
            else{
                api_faceMNN.loadModel("./model/yolo-fastestv2-opt_face.mnn");
            }
        }
    }
    else{
        ultraface.init(320, 240, 4, threshold_face);
        std::string faceModel = "";
        if (quantizationON_face){
            faceModel = "./model/slim-320-quant-ADMM-50.mnn";
            ultraface.loadModel(faceModel);
        }
        else{
            faceModel = "./model/slim-320.mnn";
            ultraface.loadModel(faceModel);
        }
    }

    bool ledON = false;
    bool cameraSeen = false;
    bool faceSeen = false;
    int prev_button = -1, current_button = -1;

    while(1){
        // if button was pressed go to wait function
        current_button = gpioRead(19);
        if(current_button == 1 && prev_button == 0){
            wait();
        }
        prev_button = current_button;

        cv::Mat frame;
        cap.read(frame);
        std::vector<TargetBox> boxes_camera;
        std::vector<TargetBox> boxes_face;
        std::vector<FaceInfo> face_info;

        Tbegin = std::chrono::steady_clock::now();
        if (useNCNNorMNN){
            api_cameraNCNN.detection(frame, boxes_camera, threshold_camera);
        }
        else{
            api_cameraMNN.detection(frame, boxes_camera, threshold_camera);
        }
        if (!useSlim320Model){
            if (useNCNNorMNN){
                api_faceNCNN.detection(frame, boxes_face, threshold_face);
            }
            else{
                api_faceMNN.detection(frame, boxes_face, threshold_face);
            }
        }
        else{
            ultraface.detect(frame, face_info);
        }
        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        std::cout << "FPS: " << f/16 << std::endl;

        if (!ledON){
            count = 0;
            // if 5 seconds passed since last detection, trigger glasses OFF
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - LED).count() >= 5){
                gpioWrite(5, PI_LOW);
                gpioWrite(26, PI_LOW);
            }
        }

        // if object was detetced in multiple frames, trigger glasses ON
        if (count >= numOfFramesToActivate && ledON){
            gpioWrite(5, PI_HIGH);
            gpioPWM(26, 128);
            std::cout << "Detected" << std::endl;
            count = 0;
            LED = std::chrono::steady_clock::now();
        }

        ledON = false;

        int baseLine = 0;
        for (int i = 0; i < boxes_camera.size(); i++) {
            cameraSeen = true;

            if (videoOutput){
                char text[256];
                sprintf(text, "%s %.1f%%", "Camera", boxes_camera[i].score * 100);

                baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = boxes_camera[i].x1;
                int y = boxes_camera[i].y1 - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > frame.cols)
                    x = frame.cols - label_size.width;

                // putting rectangles and labels on output image
                cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(255, 255, 255), -1);

                cv::putText(frame, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                cv::rectangle(frame, cv::Point(boxes_camera[i].x1, boxes_camera[i].y1),
                cv::Point(boxes_camera[i].x2, boxes_camera[i].y2), cv::Scalar(0, 255, 0), 2, 2, 0);
            }
        }

        if(!useSlim320Model){
            for (int i = 0; i < boxes_face.size(); i++) {
                faceSeen = true;

                if (videoOutput){
                    char text[256];
                    sprintf(text, "%s %.1f%%", "Face", boxes_face[i].score * 100);

                    baseLine = 0;
                    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                    int x = boxes_face[i].x1;
                    int y = boxes_face[i].y1 - label_size.height - baseLine;
                    if (y < 0)
                        y = 0;
                    if (x + label_size.width > frame.cols)
                        x = frame.cols - label_size.width;


                    cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                  cv::Scalar(255, 255, 255), -1);

                    cv::putText(frame, text, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                    cv::rectangle(frame, cv::Point(boxes_face[i].x1, boxes_face[i].y1),
                    cv::Point(boxes_face[i].x2, boxes_face[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
                }
            }
        }
        else{
            for (auto face : face_info) {
                faceSeen = true;

                if (videoOutput){
                    char text[256];
                    sprintf(text, "%s %.1f%%", "Face", face.score * 100);
                    baseLine = 0;
                    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                    int x = face.x1;
                        int y = face.y1 - label_size.height - baseLine;
                        if (y < 0)
                            y = 0;
                        if (x + label_size.width > frame.cols)
                            x = frame.cols - label_size.width;

                    // putting rectangles and labels on output image
                    cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                      cv::Scalar(255, 255, 255), -1);
                    cv::putText(frame, text, cv::Point(x, y + label_size.height),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    cv::Point pt1(face.x1, face.y1);
                    cv::Point pt2(face.x2, face.y2);
                    cv::rectangle(frame, pt1, pt2, cv::Scalar(255, 0, 255), 2);
                }
            }
        }

        if (cameraSeen && faceSeen){
            count++;
            ledON = true;
        }

        cameraSeen = false;
        faceSeen = false;

        if (videoOutput) cv::imshow("Result", frame);

        if (cv::waitKey (1) >= 0)
            break;
    }
    gpioWrite(26, PI_LOW);
    gpioTerminate(); // Compulsory.

    return 0;
}
