
# AI Recognition Program

## Introduction

The python program detects the "age", the "gender" and the "emotion" of the person in front of the laptop camera. The "age" is divided into 20 years intervals ("0-19", "20-39", "40-59", "60-79" and "80 or more"). The "gender" is divided into "male" or "female". The "emotion" are divided into 7 different emotions: "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad" and "Surprise".

The ".pdf" and ".docx" files contains the "master thesis" connected to this program. The ".jpynb" files contain the training of the neural networks for the "emotion", "age" and "gender" and the corresponding ".keras" files are the trained models. The paths for the trained data can be seen in the Jupiter files and the links to the dataset can be seen in the text files and in the "master thesis". 

## Installation

To run the Jupiter files (.ipynb) and "main.py" you will probably need to install a lot of packages in python with "pip install", see the imports that are used in the programs.

## To train the neural networks

I used "Anaconda" to train the neural networks and I used "Visual Studio Code" to run the main program that uses the laptops camera with the trained models.

1) Emotion: To train the neural network model for the "emotion" detection use the Jupiter file: "emotion-classification-cnn-using-keras.ipynb". In this file the folder paths for the "train" and "validate" data can be seen. Once the neural networks are trained, the best model (with highest accuracy) of "modelcheck_emotion.keras" (last best trained model that is saved during the modelchecks) or "model_emotion.keras" (last saved model) will be used with the "main.py" program, see also the master thesis for more information about this. This may also require to change some settings in "main.py" for which trained model that is used in the program.

2) Age: To train the neural network model for the "age" detection use the Jupiter file: "age-classification-cnn-using-keras.ipynb". In this file the folder paths for the "train" and "validate" data can be seen. Once the neural networks are trained, the best model (with highest accuracy) of "modelcheck_age.keras" (last best trained model that is saved during the modelchecks) or "model_age.keras" (last saved model) will be used with the "main.py" program, see also the master thesis for more information about this. This may also require to change some settings in "main.py" for which trained model that is used in the program.

3) Gender: To train the neural network model for the "gender" detection use the Jupiter file: "gender-classification-cnn-using-keras.ipynb". In this file the folder paths for the "train" and "validate" data can be seen. Once the neural networks are trained, the best model (with highest accuracy) of "modelcheck_gender.keras" (last best trained model that is saved during the modelchecks) or "model_gender.keras" (last saved model) will be used with the "main.py" program, see also the master thesis for more information about this. This may also require to change some settings in "main.py" for which trained model that is used in the program.

The datasets can be seen in "Dataset_Info.txt" or in the master thesis. The results of the simulations of the three kind of detections (emotion, age and gender), that are used in this project, can be seen in "Simulation_Info_Emotion.txt", "Simulation_Info_Age.txt" and "Simulation_Info_Gender.txt". 

## To run the main program

The main program that uses to the laptops camera with the trained neural network models is called "main.py", which you can run with "haarcascade_frontalface_default.xml" and the three ".keras" files (one for each of the three detection models: emotion, age and gender). All the mentioned files should be in the same folder. The computer or laptop needs to use a "web camera". Use the command (in python): "python main.py" in for instance "Visual Studio Code".

## Author

The programs and the master thesis are made by Joakim Kvistholm.
