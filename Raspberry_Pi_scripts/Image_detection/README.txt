This folder contains image detection models.

Open terminal and run:

sudo codeblocks

Once in codeblocks open this folder, and select model version folder
you would like to use. Then open the corresponding .cbp file in that
folder. The main code is located in demo.cpp file which is accessible
in codeblocks. There you can change some of the setup settings. Once
finished, simply build and run the project.

Model V1:
1 model with 3 classes: human face, phone and camera

Model V2:
2 models running in parallel:
model 1: camera
model 2: human face (custom trained or slim-320 pretrained available)

Model V3:
2 models running in parallel:
model 1: camera (improved from V2)
model 2: person

Images from All_images folder are used for detections and results are 
stored in results folder of the version used.
