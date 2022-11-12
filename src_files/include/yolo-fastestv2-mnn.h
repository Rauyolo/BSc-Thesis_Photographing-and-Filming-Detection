// this script contains parts of modified versions of scripts obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2
// https://blog.csdn.net/weixin_39266208/article/details/122131303

#ifndef YOLO_FASTEST_V2_H_
#define YOLO_FASTEST_V2_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <Interpreter.hpp>
#include "targetbox.h"

class yoloFastestv2MNN
{
private:
    std::shared_ptr<MNN::Interpreter> net = nullptr;
    MNN::Session* session = nullptr;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;

    std::vector<float> anchor;

    const char *inputName;
    const char *outputName1;
    const char *outputName2;
    const char *outputNames[2];

    int numAnchor;
    int numOutput;
    int numThreads;
    int numCategory;
    int inputWidth, inputHeight;

    float nmsThresh;

    int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes);
    int getCategory(const float *values, int index, int &category, float &score);

    int predHandle(std::unique_ptr<MNN::Tensor>*outs, std::vector<TargetBox> &dstBoxes,
                   const float scaleW, const float scaleH, const float thresh);

public:
    yoloFastestv2MNN();
    ~yoloFastestv2MNN();

    int init(int numOfClasses, std::vector<float> anchors);
    int loadModel(const char* binPath);
    int detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes,
                  const float thresh = 0.7);
};
#endif
