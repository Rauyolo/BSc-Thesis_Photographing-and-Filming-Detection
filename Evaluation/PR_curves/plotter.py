import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from numpy import trapz

# Setup
###################################################################################################################################
# select framework (NCNN or MNN)
framework = "NCNN"
#framework = "MNN"
#framework = "MNN_slim-320"

# select model version (V1 or V2) (V3 camera is same as in V2. Face class is in MNN_slim-320 framework, so use that and V3 on it)
#model_version = "V1"
#model_version = "V2"
model_version = "V3"

# use int8 for quantized version of the model, else float32
quantization = "int8"
#quantization = "float32"

# select the class for the PR curve
#PR_class = "Human_face"
#PR_class = "Mobile_phone"
#PR_class = "Camera"
PR_class = "Person"
###################################################################################################################################

PR_data_path = ""

if framework == "NCNN":
    PR_data_path = "../models/NCNN_models/" + model_version + "/results/" + PR_class + "_pr_curve_" + quantization + ".txt"
elif framework == "MNN":
    PR_data_path = "../models/MNN_models/" + model_version + "/results/" + PR_class + "_pr_curve_" + quantization + ".txt"
elif framework == "MNN_slim-320":
    PR_data_path = "../models/MNN_slim-320_model/results/" + PR_class + "_pr_curve_" + quantization + ".txt"
    
precision = []
recall = []
optimalPointFound = False
optimalPrecision = 0
optimalRecall= 0
optimalThreshold = 0
prevPrecision = 0
prevRecall = 0
prevThreshold = 0

with open(PR_data_path, 'r') as fRead:
    lines = fRead.readlines()
    for line in lines:
        line_split = line.strip().split(',')
        precision.append(float(line_split[0]))
        recall.append(float(line_split[1]))
        
        if float(line_split[0]) >= float(line_split[1]) and not optimalPointFound:
            optimalPointFound = True
            optimalPrecision = (float(line_split[0]) + prevPrecision)/2
            optimalRecall = (float(line_split[1]) + prevRecall)/2
            optimalThreshold = (float(line_split[2]) + prevThreshold)/2
        elif not optimalPointFound:
            prevPrecision = float(line_split[0])
            prevRecall = float(line_split[1])
            prevThreshold = float(line_split[2])
          
# Compute the area using the Trapezoidal Rule.
area = trapz(precision[::-1], recall[::-1])

plt.plot(recall, precision, "#fc7b0a")
plt.plot(optimalRecall, optimalPrecision, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
plt.annotate(str(round(optimalRecall, 2)) + "," + str(round(optimalPrecision, 2)), (optimalRecall + 0.015, optimalPrecision + 0.015))
plt.annotate("Opt. threshold = " + str(round(optimalThreshold, 2)), (0.695, 0.96), color = "blue")

plt.fill_between(recall,precision, color= "b", alpha= 0.2)
plt.annotate("AP = " + str(round(area, 2)), (0.3, 0.4), fontsize=20, color = "red")

plt.title("PR curve for " + PR_class + " class in Model " + model_version + ", " + framework + " " + quantization)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])

plt.savefig("./plots/" + model_version + "/" + framework + "/" + framework + "_" + quantization + "_" + PR_class +".png")

plt.show()





