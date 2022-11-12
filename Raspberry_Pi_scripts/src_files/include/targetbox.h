// this script contains script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#pragma once
class TargetBox
{
private:
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int cate;
    float score;

    float area() { return getWidth() * getHeight(); };
};
