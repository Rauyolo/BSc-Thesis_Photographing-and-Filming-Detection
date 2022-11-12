// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include "/home/pi/Desktop/src_files/include/MNN_UltraFace.hpp"
#include <sys/types.h>
#include <sys/stat.h>

// set parameters before you run
const bool useNCNNorMNN = false;        // true for NCNN, false for MNN
const bool useSlim320Model = false;     // true if slim-320, false if custom face detection model
const bool quantizationON_face = false;
const bool quantizationON_camera = false;
const float threshold_camera = 0.6;     // confidence threshold
const float threshold_face = 0.6;       // confidence threshold
const std::vector<float> anchors_camera {38.98,40.81, 77.58,94.43, 115.39,234.96, 179.64,115.53, 243.35,210.96, 284.08,313.04};
const std::vector<float> anchors_face {7.93,14.19, 18.07,31.59, 33.11,56.43, 57.30,90.38, 100.60,145.96, 200.42,230.04};

int main()
{
    system("sudo rm -r /home/pi/Desktop/Image_detection/V2/results"); // Deletes one or more files recursively.
    system("sudo mkdir /home/pi/Desktop/Image_detection/V2/results");

    // create class instances
    yoloFastestv2NCNN api_cameraNCNN;
    yoloFastestv2MNN api_cameraMNN;
    yoloFastestv2NCNN api_faceNCNN;
    yoloFastestv2MNN api_faceMNN;
    UltraFace ultraface;

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

    std::string folderpath_images = "/home/pi/Desktop/Image_detection/All_images/*.jpg";
    std::vector<std::string> filenames_images;
    cv::glob(folderpath_images, filenames_images);

    int counter = 0;
    for (size_t i=0; i<filenames_images.size(); i++){
        cv::Mat cvImg = cv::imread(filenames_images[i]);
        std::vector<TargetBox> boxes_camera;
        std::vector<TargetBox> boxes_face;
        std::vector<FaceInfo> face_info;

        if (useNCNNorMNN){
            api_cameraNCNN.detection(cvImg, boxes_camera, threshold_camera);
        }
        else{
            api_cameraMNN.detection(cvImg, boxes_camera, threshold_camera);
        }
        if (!useSlim320Model){
            if (useNCNNorMNN){
                api_faceNCNN.detection(cvImg, boxes_face, threshold_face);
            }
            else{
                api_faceMNN.detection(cvImg, boxes_face, threshold_face);
            }
        }
        else{
            ultraface.detect(cvImg, face_info);
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

        if(!useSlim320Model){
            for (int i = 0; i < boxes_face.size(); i++) {
                char text[256];
                sprintf(text, "%s %.1f%%", "Face", boxes_face[i].score * 100);

                baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = boxes_face[i].x1;
                int y = boxes_face[i].y1 - label_size.height - baseLine;
                if (y < 0)
                    y = 0;
                if (x + label_size.width > cvImg.cols)
                    x = cvImg.cols - label_size.width;


                cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(255, 255, 255), -1);

                cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                cv::rectangle(cvImg, cv::Point(boxes_face[i].x1, boxes_face[i].y1),
                cv::Point(boxes_face[i].x2, boxes_face[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
            }
        }
        else{
            for (auto face : face_info) {
                char text[256];
                sprintf(text, "%s %.1f%%", "Face", face.score * 100);
                baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                int x = face.x1;
                    int y = face.y1 - label_size.height - baseLine;
                    if (y < 0)
                        y = 0;
                    if (x + label_size.width > cvImg.cols)
                        x = cvImg.cols - label_size.width;

                // putting rectangles and labels on output image
                cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                  cv::Scalar(255, 255, 255), -1);
                cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cv::Point pt1(face.x1, face.y1);
                cv::Point pt2(face.x2, face.y2);
                cv::rectangle(cvImg, pt1, pt2, cv::Scalar(255, 0, 255), 2);
            }
        }

        cv::imwrite("/home/pi/Desktop/Image_detection/V2/results/" + std::to_string(counter) + ".jpg", cvImg);

        counter++;
    }
    return 0;
}
