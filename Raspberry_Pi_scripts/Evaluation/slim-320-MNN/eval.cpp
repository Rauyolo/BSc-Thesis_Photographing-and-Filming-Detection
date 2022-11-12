#include "src/include/MNN_UltraFace.hpp"
#include <chrono>
#include <thread>
#include <fstream>
#include <vector>

/// Fill in all of the required values before running this script
/// They can be copied from evaluation/models/MNN_slim-320_model/ModelParameters.txt
// enter location of .mnn file for MNN evaluation
const char* mnnFile = "../models/MNN_slim-320_model/slim-320-quant-ADMM-50.mnn";

const bool quantization = true;

// do not change
const static std::string class_names[] = {
    "Human_face"
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

float intersection_area(const KnownBox &a, const FaceInfo &b){
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1){
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, (int)b.x2) - std::max(a.x1, (int)b.x1);
    float inter_height = std::min(a.y2, (int)b.y2) - std::max(a.y1, (int)b.y1);

    return inter_width * inter_height;
}

float calculate_IOU(KnownBox &a, FaceInfo &b){
    if (intersection_area(a, b) != 0){
        return (intersection_area(a, b)) / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1)*(b.y2 - b.y1) - intersection_area(a, b));
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
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75,  0.80,  0.85,
    0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99,
    0.9905, 0.9910, 0.9915, 0.9920, 0.9925,
    0.9930, 0.9935, 0.9940, 0.9945, 0.9950,
    0.9955, 0.9960, 0.9965, 0.9970, 0.9975,
    0.9980, 0.9985, 0.9990, 0.9995, 0.9996,
    0.9997, 0.9998, 0.9999,
    0.99991, 0.99992, 0.99993, 0.99994, 0.99995,
    0.99996, 0.99997, 0.99998, 0.99999, 1.0
};

int main()
{
    UltraFace ultraface(mnnFile, 320, 240, 4); // config model input

    // make PR curve data
    std::string folderpath_images = "../test_images/test_set_cam_separate/Human_face/Human_face_images/*.jpg";
    std::string folderpath_labels = "../test_images/test_set_cam_separate/Human_face/Label_updated/*.txt";

    std::vector<std::string> filenames_images;
    cv::glob(folderpath_images, filenames_images);
    std::vector<std::string> filenames_labels;
    cv::glob(folderpath_labels, filenames_labels);
    std::string modelType = "";

    if (quantization){
        modelType = "int8";
    }
    else{
        modelType = "float32";
    }

    std::ofstream myfile ("../models/MNN_slim-320_model/results/Human_face_pr_curve_" + modelType + ".txt");

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
                    if (tmp_vec[0] == "Human_face"){
                        KnownBox knownBox;
                        knownBox.x1 = stoi(tmp_vec[1]);
                        knownBox.y1 = stoi(tmp_vec[2]);
                        knownBox.x2 = stoi(tmp_vec[3]);
                        knownBox.y2 = stoi(tmp_vec[4]);
                        knownBoxes.push_back(knownBox);
                    }
                }

                // run detection and store bounding boxes in target box
                std::vector<FaceInfo> face_info;
                ultraface.setScoreThreshold(thresholds[counter]);
                ultraface.detect(cvImg, face_info);

                for (int j = 0; j < knownBoxes.size(); j++) {
                    for (int k = 0; k < face_info.size(); k++){
                        float box_IOU = calculate_IOU(knownBoxes[j], face_info[k]);
                        if (box_IOU >=0.5){
                            knownBoxes[j].found = true;
                        }
                    }
                }
                // count number of TP (true positives), PP (predicted positives) and P (actual positives)
                numOfHits += countHits(knownBoxes);
                totPredBoxes += face_info.size();
                totActualBoxes += knownBoxes.size();
            }
            // precision = TP/PP
            float precision = numOfHits/(double)totPredBoxes;
            // recal = TP/P
            float recall = numOfHits/(double)totActualBoxes;
            // print to console
            std::cout << "class is: " << "Human_face" << "\n";
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
