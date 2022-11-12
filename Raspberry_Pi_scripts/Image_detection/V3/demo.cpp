// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include <chrono>
#include <pigpio.h>
#include <thread>

// set parameters before you run
const bool useNCNNorMNN = false;         // true for NCNN, false for MNN
const bool quantizationON_person = false;
const bool quantizationON_camera = false;
const bool usePhoneOnlyModel = true;
const float threshold_camera = 0.8;     // confidence threshold
const float threshold_person = 0.5;     // confidence threshold
const std::vector<float> anchors_camera {24.53,31.09, 50.78,90.88, 56.70,46.87, 108.18,97.13, 141.29,177.30, 272.16,273.26};
const std::vector<float> anchors_person {11.12,28.20, 29.20,73.18, 54.68,154.09, 109.08,249.72, 181.63,110.14, 256.38,289.73};
const std::vector<float> anchors_phone_only {22.41,23.14, 33.11,59.99, 59.28,108.37, 61.98,33.19, 113.92,61.40, 132.79,151.17};


int main()
{
    system("sudo rm -r /home/pi/Desktop/Image_detection/V3/results"); // Deletes one or more files recursively.
    system("sudo mkdir /home/pi/Desktop/Image_detection/V3/results");

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

    std::string folderpath_images = "/home/pi/Desktop/Image_detection/All_images/*.jpg";
    std::vector<std::string> filenames_images;
    cv::glob(folderpath_images, filenames_images);

    int counter = 0;
    for (size_t i=0; i<filenames_images.size(); i++){
        cv::Mat cvImg = cv::imread(filenames_images[i]);
        std::vector<TargetBox> boxes_camera;
        std::vector<TargetBox> boxes_person;

        if (useNCNNorMNN){
            api_cameraNCNN.detection(cvImg, boxes_camera, threshold_camera);
            api_personNCNN.detection(cvImg, boxes_person, threshold_person);
        }
        else{
            api_cameraMNN.detection(cvImg, boxes_camera, threshold_camera);
            api_personMNN.detection(cvImg, boxes_person, threshold_person);
        }

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
            if (x + label_size.width > cvImg.cols)
                x = cvImg.cols - label_size.width;

            // putting rectangles and labels on output image
            cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            cv::rectangle(cvImg, cv::Point(boxes_camera[i].x1, boxes_camera[i].y1),
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
            if (x + label_size.width > cvImg.cols)
                x = cvImg.cols - label_size.width;


            cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            cv::rectangle(cvImg, cv::Point(boxes_person[i].x1, boxes_person[i].y1),
            cv::Point(boxes_person[i].x2, boxes_person[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
        }

        cv::imwrite("/home/pi/Desktop/Image_detection/V3/results/" + std::to_string(counter) + ".jpg", cvImg);

        counter++;
    }

    return 0;
}
