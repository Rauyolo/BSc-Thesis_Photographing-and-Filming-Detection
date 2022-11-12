// this script contains parts of a modified version of script obtained from:
// https://github.com/dog-qiuqiu/Yolo-FastestV2

#include "src/include/yolo-fastestv2.h"
#include <chrono>
#include <thread>
#include <fstream>
#include <vector>

/// Fill in all of the required values before running this script
/// They can be copied from evaluation/models/MNN_models/V<your_version>/ModelParameters.txt
// enter location of .mnn file for MNN evaluation
const char* mnnFile = "../models/MNN_models/V3/yolo-fastestv2-int8_person.mnn";

// folder where resulting PR curve data will be stored
const char* destPath = "../models/MNN_models/V3/results/";

const bool quantization = true;

// enter anchor bias corresponding to the selected model
const std::vector<float> anchors {11.12,28.20, 29.20,73.18, 54.68,154.09, 109.08,249.72, 181.63,110.14, 256.38,289.73};

// true if camera and mobile phone class were combined in one in the model
const bool camAndPhoneCombined = true;

// order matters, use same as in the training
const static std::string class_names[] = {
    "Person"
};
/// Done

class KnownBox{
public:
    int x1;
    int y1;
    int x2;
    int y2;
    bool found = false;
};

float intersection_area(const KnownBox &a, const TargetBox &b){
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1){
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

float calculate_IOU(KnownBox &a, TargetBox &b){
    if (intersection_area(a, b) != 0){
        return (intersection_area(a, b)) / ((a.x2 - a.x1) * (a.y2 - a.y1) + b.area() - intersection_area(a, b));
    }
    else{
        return 0;
    }
}

int countHits(std::vector<KnownBox> &knownBoxes){
    int hitCounter = 0;
    for (auto const& box : knownBoxes) {
        if (box.found){
            hitCounter++;
        }
    }
    return hitCounter;
}

std::vector<float> thresholds {
    0.0, 0.0002, 0.0004, 0.0006, 0.0008,
    0.0010, 0.0012, 0.0014, 0.0016, 0.0018,
    0.0020, 0.0022, 0.0024, 0.0026, 0.0028,
    0.0030, 0.0032, 0.0034, 0.0036, 0.0038,
    0.0040, 0.0042, 0.0044, 0.0046, 0.0048,
    0.0050, 0.0052, 0.0054, 0.0056, 0.0058,
    0.0060, 0.0062, 0.0064, 0.0066, 0.0068,
    0.0070, 0.0072, 0.0074, 0.0076, 0.0078,
    0.0080, 0.0082, 0.0084, 0.0086, 0.0088,
    0.0090, 0.0092, 0.0094, 0.0096, 0.0098,
    0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
    0.040, 0.045, 0.050, 0.055, 0.060, 0.065,
    0.070, 0.075, 0.080, 0.085, 0.090, 0.095,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94,
    0.95, 0.96, 0.97, 0.98, 0.99,
    0.9905, 0.9910, 0.9915, 0.9920, 0.9925,
    0.9930, 0.9935, 0.9940, 0.9945, 0.9950,
    0.9955, 0.9960, 0.9965, 0.9970, 0.9975,
    0.9980, 0.9985, 0.9990, 0.9995, 0.9996,
    0.9997, 0.9998, 0.9999,
    0.99991, 0.99992, 0.99993, 0.99994, 0.99995,
    0.99996, 0.99997, 0.99998, 0.99999,
    0.999991, 0.999992, 0.999993, 0.999994, 0.999995,
    0.999996, 0.999997, 0.999998,0.999999, 1.0
};

int main()
{
    yoloFastestv2 api(sizeof(class_names)/sizeof(*class_names), anchors);
    api.loadModel(mnnFile);

    // make PR curve data for each class
    for ( auto& class_name : class_names )
    {
        std::string folderpath_images = "";
        std::string folderpath_labels = "";

        // opens folder with images. for more classes, images have to be manually added to these folders
        if (camAndPhoneCombined){
            folderpath_images = "../test_images/test_set_cam_combined/" + class_name + "/" + class_name + "_images/*.jpg";
            folderpath_labels = "../test_images/test_set_cam_combined/" + class_name + "/" + "Label_updated/*.txt";
        }
        else{
            folderpath_images = "../test_images/test_set_cam_separate/" + class_name + "/" + class_name + "_images/*.jpg";
            folderpath_labels = "../test_images/test_set_cam_separate/" + class_name + "/" + "Label_updated/*.txt";
        }

        std::vector<std::string> filenames_images;
        cv::glob(folderpath_images, filenames_images);
        std::vector<std::string> filenames_labels;
        cv::glob(folderpath_labels, filenames_labels);
        char* modelType = "";

        if (quantization){
            modelType = "int8";
        }
        else{
            modelType = "float32";
        }

        std::ofstream myfile (destPath + class_name + "_pr_curve_" + modelType + ".txt");

        if (myfile.is_open())
        {
            int counter = 0;
            // loop through different values of threshold for PR curve data
            while(true){
                int numOfHits = 0;
                int totPredBoxes = 0;
                int totActualBoxes = 0;
                // loop through images in the class folder
                for (size_t i=0; i<filenames_images.size(); i++)
                {
                    // read the image
                    cv::Mat cvImg = cv::imread(filenames_images[i]);
                    std::string line = "";
                    std::ifstream inFile;
                    inFile.open(filenames_labels[i]);

                    if (!inFile) {
                        std::cout << "Unable to open text file";
                        exit(1); // terminate with error
                    }

                    // create the known bounding boxes of the class objects in the image
                    std::vector<KnownBox> knownBoxes;
                    while (getline(inFile, line)) {
                        std::string tmp;
                        std::stringstream ss(line);
                        std::vector<std::string> tmp_vec;
                        while(getline(ss, tmp, ' ')){
                            tmp_vec.push_back(tmp);
                        }
                        if (tmp_vec[0] == class_name){
                            KnownBox knownBox;
                            knownBox.x1 = stoi(tmp_vec[1]);
                            knownBox.y1 = stoi(tmp_vec[2]);
                            knownBox.x2 = stoi(tmp_vec[3]);
                            knownBox.y2 = stoi(tmp_vec[4]);
                            knownBoxes.push_back(knownBox);
                        }
                    }

                    // run detection and store bounding boxes in target box
                    std::vector<TargetBox> boxes;
                    api.detection(cvImg, boxes, thresholds[counter]);

                    for (int j = 0; j < knownBoxes.size(); j++) {
                        for (int k = 0; k < boxes.size(); k++){
                            if (class_names[boxes[k].cate] == class_name){
                                float box_IOU = calculate_IOU(knownBoxes[j], boxes[k]);
                                if (box_IOU >=0.5){
                                    knownBoxes[j].found = true;
                                }
                            }
                        }
                    }

                    for (int k = 0; k < boxes.size(); k++){
                            if (class_names[boxes[k].cate] == class_name){
                                // count number of PP (predicted positives)
                                totPredBoxes++;
                            }
                    }
                    // count number of TP (true positives) and P (actual positives)
                    numOfHits += countHits(knownBoxes);
                    totActualBoxes += knownBoxes.size();
                }
                // precision = TP/PP
                float precision = numOfHits/(double)totPredBoxes;
                // recal = TP/P
                float recall = numOfHits/(double)totActualBoxes;
                // print to console
                std::cout << "class is: " << class_name << "\n";
                std::cout << "For threshold value of " << thresholds[counter] << " the following was found:"<< std::endl;
                std::cout << "precision is: " << precision << std::endl;
                std::cout << "recall is: " << recall << std::endl << std::endl;

                // write precision and recall to the corresponding file
                if (precision != precision)     // check if NaN
                    myfile << "1" << "," << recall << ","<< thresholds[counter] << "\n";
                else
                    myfile << precision << "," << recall << ","<< thresholds[counter] << "\n";

                counter++;
                if (counter == thresholds.size()){
                    break;
                }
            }

            myfile.close();
        }
        else std::cout << "Unable to open file";
    }
}
