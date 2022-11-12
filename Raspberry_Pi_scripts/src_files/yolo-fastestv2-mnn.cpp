// this script contains parts of modified versions of scripts obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2
// https://blog.csdn.net/weixin_39266208/article/details/122131303

#include <cmath>
#include <algorithm>
#include "include/yolo-fastestv2-mnn.h"
#include "Tensor.hpp"
#include <opencv2/core/matx.hpp>
#include <vector>
#include <memory>

using namespace std;

yoloFastestv2MNN::yoloFastestv2MNN()
{

}

yoloFastestv2MNN::~yoloFastestv2MNN()
{
    printf("Destroying yoloFastestv2 Detector...\n");
}

//parameter configuration of the model
int yoloFastestv2MNN::init(int numOfClasses, std::vector<float> anchors){
    printf("Creating yoloFastestv2 Detector...\n");
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
    inputName = "input.1";
    outputNames[0] = "794"; //22x22
    outputNames[1] = "796"; //11x11
    outputName1 = "794"; //22x22
    outputName2 = "796"; //11x11

    //print initialization related information
    printf("numThreads:%d\n", numThreads);
    printf("inputWidth:%d inputHeight:%d\n", inputWidth, inputHeight);

    std::vector<float> bias = anchors;
    anchor.assign(bias.begin(), bias.end());

    return 0;
}

//mnn model loading
int yoloFastestv2MNN::loadModel(const char* path)
{
    printf("MNN mode init:\n%s\n%s\n", path);

    net = std::shared_ptr<MNN::Interpreter> (MNN::Interpreter::createFromFile(path));
    config.numThread = numThreads;
    config.type = MNN_FORWARD_CPU;

#if 0
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
#endif
    session = net->createSession(config);

    printf("MNN model init success...\n");

    return 0;
}

float intersection_area_MNN(const TargetBox &a, const TargetBox &b)
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

bool scoreSort_MNN(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

//NMS processing
int yoloFastestv2MNN::nmsHandle(std::vector<TargetBox> &tmpBoxes,
                             std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;

    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort_MNN);

    for (int i = 0; i < tmpBoxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //intersection
            float inter_area = intersection_area_MNN(tmpBoxes[i], tmpBoxes[picked[j]]);
            //union
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if(IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate) {
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
int yoloFastestv2MNN::getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore  = values[4 * numAnchor + index];

#if 1
    auto start = &values[5*numAnchor];
    auto end = &values[5*numAnchor] + numCategory;
    category = std::max_element(start, end) - start;
    score = start[category] * objScore;

#else
    for (int i = 0; i < numCategory; i++) {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if(clsScore > tmp) {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }
#endif

    return 0;
}

//feature map post-processing
int yoloFastestv2MNN::predHandle(std::unique_ptr<MNN::Tensor>*outs, std::vector<TargetBox> &dstBoxes,
                              const float scaleW, const float scaleH, const float thresh)
{    //do result
    for (int i = 0; i < numOutput; i++) {
        int stride;
        int outW, outH, outC;
        auto &out = outs[i];
        auto shape = out->shape();

        outH = shape[1];
        outW = shape[2];
        outC = shape[3];

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        auto values = out->host<float>();

        for (int h = 0; h < outH; h++) {
            const float* valueh = &values[h*outW*outC];

            for (int w = 0; w < outW; w++) {
                for (int b = 0; b < numAnchor; b++) {
                    //float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(valueh, b, category, score);

                    if (score > thresh) {
                        float bcx, bcy, bw, bh;

                        bcx = ((valueh[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((valueh[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((valueh[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((valueh[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                valueh += outC;
            }
        }
    }
    return 0;
}

int yoloFastestv2MNN::detection(const cv::Mat srcImg, std::vector<TargetBox> &dstBoxes, const float thresh)
{
    dstBoxes.clear();

    float scaleW = (float)srcImg.cols / (float)inputWidth;
    float scaleH = (float)srcImg.rows / (float)inputHeight;

    cv::Mat small;
    cv::resize(srcImg, small, cv::Size(), 1./scaleW, 1./scaleH, cv::INTER_LINEAR);
    small.convertTo(small, CV_32FC3, 1./255);

    auto input = net->getSessionInput(session, NULL);
    std::vector<int> dim{1, inputHeight, inputWidth, 3};

    std::unique_ptr<MNN::Tensor> nhwc_Tensor(MNN::Tensor::create<float>(dim, NULL, MNN::Tensor::TENSORFLOW));
    auto nhwc_data = nhwc_Tensor->host<float>();
    auto nhwc_size = nhwc_Tensor->size();
    ::memcpy(nhwc_data, small.data, nhwc_size);
    input->copyFromHostTensor(nhwc_Tensor.get());

    net->runSession(session);

    auto outmap = net->getSessionOutputAll(session);
    std::unique_ptr<MNN::Tensor> out[2];

    out[0] = std::make_unique<MNN::Tensor>(outmap[outputNames[0]], MNN::Tensor::CAFFE);
    out[1] = std::make_unique<MNN::Tensor>(outmap[outputNames[1]], MNN::Tensor::CAFFE);

    outmap[outputNames[0]]->copyToHostTensor(out[0].get());
    outmap[outputNames[1]]->copyToHostTensor(out[1].get());

#if 0
    auto shape = outmap[outputNames[0]]->shape();
    auto val0 = out[0]->host<float>();
    auto val1 = outmap[outputNames[0]]->host<float>();
    for(int i = 0; i < 10; i++){
        std::cout << val0[i] << " " << val1[i] << std::endl;
    }
#endif

    std::vector<TargetBox> tmpBoxes;
    //feature map post-processing
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);

    //NMS
    nmsHandle(tmpBoxes, dstBoxes);

    return 0;
}
