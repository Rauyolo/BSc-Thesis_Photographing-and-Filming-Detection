// modified from https://github.com/dog-qiuqiu/Yolo-FastestV2

#define _CRT_SECURE_NO_WARNINGS
#define _NO_CRT_STDIO_INLINE
#include "src/yolo-fastestv2.h"
#include <chrono>
#include <thread>

// set parameters before you run
const bool quantizationON = false;
const float threshold_camera = 0.6;
const float threshold_person = 0.7;
const std::vector<float> anchors_camera{ 24.53,31.09, 50.78,90.88, 56.70,46.87, 108.18,97.13, 141.29,177.30, 272.16,273.26 };
const std::vector<float> anchors_person{ 11.12,28.20, 29.20,73.18, 54.68,154.09, 109.08,249.72, 181.63,110.14, 256.38,289.73 };

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    float f;
    float FPS[16];
    int i, Fcnt = 0;
    std::chrono::steady_clock::time_point Tbegin, Tend;

    for (i = 0; i < 16; i++) FPS[i] = 0.0;

    // create yoloFastestv2 class instances
    yoloFastestv2 api_camera(1, anchors_camera);
    yoloFastestv2 api_person(1, anchors_person);

    if (quantizationON) {
        api_camera.loadModel("./model/yolo-fastestv2-int8_camera.param",
            "./model/yolo-fastestv2-int8_camera.bin");
    }
    else {
        api_camera.loadModel("./model/yolo-fastestv2-opt_camera.param",
            "./model/yolo-fastestv2-opt_camera.bin");
    }

    if (quantizationON) {
        api_person.loadModel("./model/yolo-fastestv2-int8_person.param",
            "./model/yolo-fastestv2-int8_person.bin");
    }
    else {
        api_person.loadModel("./model/yolo-fastestv2-opt_person.param",
            "./model/yolo-fastestv2-opt_person.bin");
    }

    while (1) {
        cv::Mat frame;
        cap.read(frame);
        std::vector<TargetBox> boxes_camera;
        std::vector<TargetBox> boxes_person;
        Tbegin = std::chrono::steady_clock::now();
        api_camera.detection(frame, boxes_camera, threshold_camera);
        api_person.detection(frame, boxes_person, threshold_person);
        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
        for (f = 0.0, i = 0; i < 16; i++) { f += FPS[i]; }
        std::cout << "FPS: " << f / 16 << std::endl;

        int baseLine = 0;
        for (int i = 0; i < boxes_camera.size(); i++) {
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

        for (int i = 0; i < boxes_person.size(); i++) {
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

        cv::imshow("Result", frame);

        if (cv::waitKey(1) >= 0)
            break;
    }

    return 0;
}
