// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-ncnn.h"
#include "/home/pi/Desktop/src_files/include/yolo-fastestv2-mnn.h"
#include <sys/types.h>
#include <sys/stat.h>

const bool useNCNNorMNN = true;        // true for NCNN, false for MNN
const bool quantizationON = false;
const float threshold = 0.5;            // confidence threshold
const std::vector<float> anchors {11.52,19.71, 34.37,52.14, 77.47,96.86, 117.81,223.21, 197.90,131.73, 274.62,287.37};
static const char* class_names[] = {
    "Face", "Phone", "Camera"
};

int main()
{
    system("sudo rm -r /home/pi/Desktop/Image_detection/V1/results"); // Deletes one or more files recursively.
    system("sudo mkdir /home/pi/Desktop/Image_detection/V1/results");

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

    std::string folderpath_images = "/home/pi/Desktop/Image_detection/All_images/*.jpg";
    std::vector<std::string> filenames_images;
    cv::glob(folderpath_images, filenames_images);

    int counter = 0;
    for (size_t i=0; i<filenames_images.size(); i++){
        cv::Mat cvImg = cv::imread(filenames_images[i]);
        std::vector<TargetBox> boxes;
        if (useNCNNorMNN){
            api_NCNN.detection(cvImg, boxes, threshold);
        }
        else{
            api_MNN.detection(cvImg, boxes, threshold);
        }

        for (int i = 0; i < boxes.size(); i++) {
            char text[256];
            sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = boxes[i].x1;
            int y = boxes[i].y1 - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > cvImg.cols)
                x = cvImg.cols - label_size.width;

            cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            if (boxes[i].cate == 0) {
                cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 0, 255), 2, 2, 0);
            }
            else if (boxes[i].cate == 1) {
                cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(0, 255, 0), 2, 2, 0);
            }
            else if (boxes[i].cate == 2) {
                cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(0, 0, 255), 2, 2, 0);
            }
        }

        cv::imwrite("/home/pi/Desktop/Image_detection/V1/results/" + std::to_string(counter) + ".jpg", cvImg);

        counter++;
    }
    return 0;
}
