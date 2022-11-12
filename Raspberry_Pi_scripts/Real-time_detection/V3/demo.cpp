// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include <chrono>
#include <pigpio.h>
#include <thread>

// set parameters before you run
const bool videoOutput = false;
const bool useNCNNorMNN = false;         // true for NCNN, false for MNN
const bool quantizationON_person = false;
const bool quantizationON_camera = false;
const bool usePhoneOnlyModel = true;
const float threshold_camera = 0.6;     // confidence threshold
const float threshold_person = 0.7;     // confidence threshold
const int numOfFramesToActivate = 5;    // set how many frames need to be detected, before glasses turn dark
const float area_of_overlap = 0.7;      // set how much area of camera box needs to overlap with human box (0 to 1)
const std::vector<float> anchors_camera {24.53,31.09, 50.78,90.88, 56.70,46.87, 108.18,97.13, 141.29,177.30, 272.16,273.26};
const std::vector<float> anchors_person {11.12,28.20, 29.20,73.18, 54.68,154.09, 109.08,249.72, 181.63,110.14, 256.38,289.73};
const std::vector<float> anchors_phone_only {22.41,23.14, 33.11,59.99, 59.28,108.37, 61.98,33.19, 113.92,61.40, 132.79,151.17};

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

bool intersection(std::vector<TargetBox> &camVec, std::vector<TargetBox> &personVec){
    for (auto& cam : camVec){
        for (auto& person : personVec){
            if (cam.x1 > person.x2 || cam.x2 < person.x1 || cam.y1 > person.y2 || cam.y2 < person.y1){
                // no intersection
                continue;
            }
            else{
                float inter_width = std::min(cam.x2, person.x2) - std::max(cam.x1, person.x1);
                float inter_height = std::min(cam.y2, person.y2) - std::max(cam.y1, person.y1);
                if ((inter_width * inter_height) / cam.area() >= area_of_overlap){
                    return true;
                }
                else{
                    continue;
                }
            }
        }
    }
    return false;
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
    yoloFastestv2NCNN api_personNCNN;
    yoloFastestv2MNN api_personMNN;

    if (useNCNNorMNN){
        if (!usePhoneOnlyModel){
            api_cameraNCNN.init(1, anchors_camera);
        }
        else{
            api_cameraNCNN.init(1, anchors_phone_only);
        }
        api_personNCNN.init(1, anchors_person);

        if (quantizationON_camera){
            if (!usePhoneOnlyModel){
                api_cameraNCNN.loadModel("./model/yolo-fastestv2-int8_camera.param",
                      "./model/yolo-fastestv2-int8_camera.bin");
            }
            else{
                api_cameraNCNN.loadModel("./model/yolo-fastestv2-int8_phone_only.param",
                    "./model/yolo-fastestv2-int8_phone_only.bin");
            }
        }
        else{
            if (!usePhoneOnlyModel){
                api_cameraNCNN.loadModel("./model/yolo-fastestv2-opt_camera.param",
                      "./model/yolo-fastestv2-opt_camera.bin");
            }
            else{
                api_cameraNCNN.loadModel("./model/yolo-fastestv2-opt_phone_only.param",
                      "./model/yolo-fastestv2-opt_phone_only.bin");
            }
        }

        if (quantizationON_person){
            api_personNCNN.loadModel("./model/yolo-fastestv2-int8_person.param",
                      "./model/yolo-fastestv2-int8_person.bin");
        }
        else{
            api_personNCNN.loadModel("./model/yolo-fastestv2-opt_person.param",
                      "./model/yolo-fastestv2-opt_person.bin");
        }
    }
    else{
        if (!usePhoneOnlyModel){
            api_cameraMNN.init(1, anchors_camera);
        }
        else{
            api_cameraMNN.init(1, anchors_phone_only);
        }
        api_personMNN.init(1, anchors_person);

        if (quantizationON_camera){
            if (!usePhoneOnlyModel){
                api_cameraMNN.loadModel("./model/yolo-fastestv2-int8_camera.mnn");
            }
            else{
                api_cameraMNN.loadModel("./model/yolo-fastestv2-int8_phone_only.mnn");
            }
        }
        else{
            if (!usePhoneOnlyModel){
                api_cameraMNN.loadModel("./model/yolo-fastestv2-opt_camera.mnn");
            }
            else{
                api_cameraMNN.loadModel("./model/yolo-fastestv2-opt_phone_only.mnn");
            }
        }

        if (quantizationON_person){
            api_personMNN.loadModel("./model/yolo-fastestv2-int8_person.mnn");
        }
        else{
            api_personMNN.loadModel("./model/yolo-fastestv2-opt_person.mnn");
        }
    }

    bool ledON = false;
    bool cameraSeen = false;
    bool personSeen = false;
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
        std::vector<TargetBox> boxes_person;

        Tbegin = std::chrono::steady_clock::now();
        if (useNCNNorMNN){
            api_cameraNCNN.detection(frame, boxes_camera, threshold_camera);
            api_personNCNN.detection(frame, boxes_person, threshold_person);
        }
        else{
            api_cameraMNN.detection(frame, boxes_camera, threshold_camera);
            api_personMNN.detection(frame, boxes_person, threshold_person);
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


        for (int i = 0; i < boxes_person.size(); i++) {
            personSeen = true;

            if (videoOutput){
                char text[256];
                sprintf(text, "%s %.1f%%", "Person", boxes_person[i].score * 100);

                baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = boxes_person[i].x1;
                int y = boxes_person[i].y1 - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > frame.cols)
                    x = frame.cols - label_size.width;


                cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(255, 255, 255), -1);

                cv::putText(frame, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                cv::rectangle(frame, cv::Point(boxes_person[i].x1, boxes_person[i].y1),
                cv::Point(boxes_person[i].x2, boxes_person[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
            }
        }

        if (cameraSeen && personSeen && intersection(boxes_camera, boxes_person)){
            count++;
            ledON = true;
        }

        cameraSeen = false;
        personSeen = false;

        if (videoOutput) cv::imshow("Result", frame);

        if (cv::waitKey (1) >= 0)
            break;
    }
    gpioWrite(26, PI_LOW);
    gpioTerminate(); // Compulsory.

    return 0;
}
