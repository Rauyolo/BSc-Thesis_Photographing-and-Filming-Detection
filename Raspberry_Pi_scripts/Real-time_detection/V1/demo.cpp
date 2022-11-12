// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include <chrono>
#include <pigpio.h>
#include <thread>

// set parameters before you run
const bool videoOutput = false;
const bool useNCNNorMNN = true;         // true for NCNN, false for MNN
const bool quantizationON = false;
const float threshold = 0.5;            // confidence threshold
const int numOfFramesToActivate = 10;   // set how many frames need to be detected, before glasses turn dark
const std::vector<float> anchors {11.52,19.71, 34.37,52.14, 77.47,96.86, 117.81,223.21, 197.90,131.73, 274.62,287.37};
static const char* class_names[] = {
    "Face", "Phone", "Camera"
};

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

    // create yoloFastestv2 class instance
    yoloFastestv2NCNN api_NCNN;
    yoloFastestv2MNN api_MNN;

    if (useNCNNorMNN){
        api_NCNN.init(sizeof(class_names)/sizeof(*class_names), anchors);
        if (quantizationON){
            api_NCNN.loadModel("./model/yolo-fastestv2-int8.param",
                      "./model/yolo-fastestv2-int8.bin");
        }
        else{
            api_NCNN.loadModel("./model/yolo-fastestv2-opt.param",
                      "./model/yolo-fastestv2-opt.bin");
        }
    }
    else{
        api_MNN.init(sizeof(class_names)/sizeof(*class_names), anchors);
        if (quantizationON){
            api_MNN.loadModel("./model/yolo-fastestv2_int8.mnn");
        }
        else{
            api_MNN.loadModel("./model/yolo-fastestv2.mnn");
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
        std::vector<TargetBox> boxes;
        Tbegin = std::chrono::steady_clock::now();
        if (useNCNNorMNN){
            api_NCNN.detection(frame, boxes, threshold);
        }
        else{
            api_MNN.detection(frame, boxes, threshold);
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
        for (int i = 0; i < boxes.size(); i++) {
            if(boxes[i].cate == 0){
                faceSeen = true;
            }
            else if(boxes[i].cate == 1 || boxes[i].cate == 2){
                cameraSeen = true;
            }

            if(faceSeen && cameraSeen){
                count++;
                ledON = true;
            }
            if (videoOutput){
                char text[256];
                sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);

                baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = boxes[i].x1;
                int y = boxes[i].y1 - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > frame.cols)
                    x = frame.cols - label_size.width;

                // putting rectangles and labels on output image
                cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(255, 255, 255), -1);

                cv::putText(frame, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                if (boxes[i].cate == 0) {
                    cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
                        cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
                }
                else if (boxes[i].cate == 1) {
                    cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
                        cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(0, 255, 0), 2, 2, 0);
                }
                else if (boxes[i].cate == 2) {
                    cv::rectangle(frame, cv::Point(boxes[i].x1, boxes[i].y1),
                        cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(0, 0, 255), 2, 2, 0);
                }
            }
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
