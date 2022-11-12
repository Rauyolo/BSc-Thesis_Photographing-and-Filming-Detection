This folder contains tools to evaluate models.

Open terminal and run:

sudo codeblocks

Once in codeblocks open this folder, select a folder for framework you
would like to evaluate: NCNN, MNN or slim-320 face model. Then open 
the corresponding .cbp file in that folder. The main code is located 
in eval.cpp file which is accessible in codeblocks. There you can 
change some of the setup settings. All models are located in 
/home/pi/Desktop/Evaluation/models. Once you select your model framework
and version there, you can use ModelParameters.txt file and copy the 
setup code to evaluate this model. Go back to eval.cpp and replace
setup code with the one you copied (the setup explains it too). Once
finished, simply build and run the project. It might take a while to 
evaluate the model at multiple thresholds, so please be patient.

Once done, a text file will be generated in 
/home/pi/Desktop/Evaluation/models/<your framework>/<your model>/results

Go to /home/pi/Desktop/Evaluation/PR_curves/plotter.py and there 
comment/uncomment lines to get lines needed to evaluate your model.
A plot will be generated and displayed. They are also saved in 
/home/pi/Desktop/Evaluation/PR_curves/plots directory.
