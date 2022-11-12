// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#ifndef NCNN_H_
#define NCNN_H_

#include "net.h"

#include <vector>
#include <opencv2/opencv.hpp>
#include "targetbox.h"

class yoloFastestv2NCNN
{
private:
    ncnn::Net net;
    std::vector<float> anchor;

    char *inputName;
    char *outputName1;
    char *outputName2;

    int numAnchor;
    int numOutput;
    int numThreads;
    int numCategory;
    int inputWidth, inputHeight;

    float nmsThresh;

    int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes);
    int getCategory(const float *values, int index, int &category, float &score);
    int predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes,
                   const float scaleW, const float scaleH, const float thresh);

public:
    yoloFastestv2NCNN();
    ~yoloFastestv2NCNN();

    int init(int numOfClasses, std::vector<float> anchors);
    int loadModel(const char* paramPath, const char* binPath);
    int detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes,
                  const float thresh = 0.3);
};
#endif
