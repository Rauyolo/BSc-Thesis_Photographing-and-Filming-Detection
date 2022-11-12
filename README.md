# Photographing and Filming Detection
<p align="center">
<img src="examples.png" width="500" height="500">


This repository includes scripts for the training and deployment of a photography and filming detection system. The training and NCNN conversion is done via **YOLO_notebook.ipynb** file. Due to copyright issues, the custom dataset that was created for the final model is not available in this repository and is available on request. The deployment is done on Raspberry Pi 4, which has OpenCV, NCNN, MNN, and pigpio libraries built and the scripts used on it are given in **Raspberry_Pi_scripts** folder. 

It is highly advised to use the Raspberry Pi image available [here](https://drive.google.com/file/d/1YTJM-GwtlU87NoIdmZxIRFkDgh-7eCbv/view?usp=sharing) with the following credentials:
- Username: **pi**
- Password: **3.14**

This image already contains all the needed libraries and scripts for deployment and evaluation of NCNN and MNN models. It also has 16GB of available space.

## YOLO_notebook.ipynb
This notebook contains 11 different sections with which you can download needed images, label them, perform data augmentation, train the model, evaluate it using image or real-time detection, and deploy it on NCNN. Most sections can be run separately, but be sure to stick with the required folder structure of **YOLO_notebook.ipynb**. To not have issues with some sections not runnning, it is advised to run through the whole notebook at least once so that all required directories are created.

## Yolo-FastestV2-main folder
This folder includes most of the directories used by **YOLO_notebook.ipynb**. 

**Yolo-FastestV2-main/data** contains **.names** file, which contains names of classes in the current model and **.data** file, which contains information about the model (number of classes, anchors and more). Both files are used during training and model evaluation. If you wish to use available **.data** files, make sure your model name is the same. 

**Yolo-FastestV2-main/model** directory includes YoloFastestV2 model and should not be changed. 

**Yolo-FastestV2-main/modelzoo** contains weight files of the final models you wish to evaluate or deploy. 

**Yolo-FastestV2-main/results** includes two folders: in **input_img**, you can put all of the images you wish to evaluate, and by running section 9.2 of the notebook, output images will be generated in **output_img** folder.

When you convert a model to NCNN, it will be saved in **Yolo-FastestV2-main/sample/ncnn/model** folder. If you wish to deploy this model on Windows, follow the instructions given in section 10.6 of the notebook. Otherwise, the files can be sent to Raspberry Pi and used there.

**Yolo-FastestV2-main/utils** folder includes scripts used by **train.py** and **evaluation.py** files.

**Yolo-FastestV2-main/weights** folder includes weight files of the trained models saved every ten epochs.

Please refer to [YoloFastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2) repository for more information.

