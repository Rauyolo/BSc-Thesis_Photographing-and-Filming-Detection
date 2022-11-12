#include <math.h>
#include <algorithm>
#include "yolo-fastestv2.h"

//parameter configuration of the model
yoloFastestv2::yoloFastestv2(int numOfClasses, std::vector<float> anchors)
{
    printf("Creat yoloFastestv2 Detector...\n");
    //number of output nodes
    numOutput = 2;
    //number of inference threads
    numThreads = 4;
    //anchor num
    numAnchor = 3;
    //number of categories
    numCategory = numOfClasses;
    //NMS threshold
    nmsThresh = 0.25;

    //model input size
    inputWidth = 352;
    inputHeight = 352;

    //model input and output node names
    const char* inputNameA = "input.1";
    inputName = NULL;
    inputName = (char*)inputNameA;

    const char* outputName1A = "794";
    outputName1 = NULL;
    outputName1 = (char*)outputName1A;

    const char* outputName2A = "796";
    outputName2 = NULL;
    outputName2 = (char*)outputName2A;

    //print initialization related information
    printf("numThreads:%d\n", numThreads);
    printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

    //anchor box w h
    std::vector<float> bias = anchors;
    anchor.assign(bias.begin(), bias.end());

}

yoloFastestv2::~yoloFastestv2()
{
    printf("Destroy yoloFastestv2 Detector...\n");
}

//ncnn model loading
int yoloFastestv2::loadModel(const char* paramPath, const char* binPath)
{
    printf("Ncnn mode init:\n%s\n%s\n", paramPath, binPath);

    net.load_param(paramPath);
    net.load_model(binPath);

    printf("Ncnn model init sucess...\n");

    return 0;
}

float intersection_area(const TargetBox& a, const TargetBox& b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

//NMS processing
int yoloFastestv2::nmsHandle(std::vector<TargetBox>& tmpBoxes,
    std::vector<TargetBox>& dstBoxes)
{
    std::vector<int> picked;

    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //intersection
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            //union
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++) {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

//detection class score processing
int yoloFastestv2::getCategory(const float* values, int index, int& category, float& score)
{
    float tmp = 0;
    float objScore = values[4 * numAnchor + index];

    for (int i = 0; i < numCategory; i++) {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if (clsScore > tmp) {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }

    return 0;
}

//feature map post-processing
int yoloFastestv2::predHandle(const ncnn::Mat* out, std::vector<TargetBox>& dstBoxes,
    const float scaleW, const float scaleH, const float thresh)
{    //do result
    for (int i = 0; i < numOutput; i++) {
        int stride;
        int outW, outH, outC;

        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; h++) {
            const float* values = out[i].channel(h);

            for (int w = 0; w < outW; w++) {
                for (int b = 0; b < numAnchor; b++) {
                    //float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh) {
                        float bcx, bcy, bw, bh;

                        bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;
            }
        }
    }
    return 0;
}

int yoloFastestv2::detection(const cv::Mat srcImg, std::vector<TargetBox>& dstBoxes, const float thresh)
{
    dstBoxes.clear();

    float scaleW = (float)srcImg.cols / (float)inputWidth;
    float scaleH = (float)srcImg.rows / (float)inputHeight;

    //resize of input image data
    ncnn::Mat inputImg = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR, \
        srcImg.cols, srcImg.rows, inputWidth, inputHeight);

    //Normalization of input image data
    const float mean_vals[3] = { 0.f, 0.f, 0.f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    inputImg.substract_mean_normalize(mean_vals, norm_vals);

    //creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(numThreads);

    //set input tensor
    ex.input(inputName, inputImg);

    //forward
    ncnn::Mat out[2];
    ex.extract(outputName1, out[0]); //22x22
    ex.extract(outputName2, out[1]); //11x11

    std::vector<TargetBox> tmpBoxes;
    //feature map post-processing
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    //NMS
    nmsHandle(tmpBoxes, dstBoxes);

    return 0;
}
