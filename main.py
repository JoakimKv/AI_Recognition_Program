
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Gets the correct model to load from the trained neural network (that predicts the "emotion", "age" or "gender"). 
# The "True" alternative is the model from the "best epoch" and the "False" alternative is from the "last saved"
# model (this is chosen from all three models).
def getCorrectModelsToLoad(isBestModelEmotion, isBestModelAge, isBestModelGender):

    model_vect_local = []
      
    if (isBestModelEmotion == True):      
       model_vect_local.append(f"modelcheck_emotion.keras")
    else:       
       model_vect_local.append(f"model_emotion.keras")    

    if (isBestModelAge == True):      
       model_vect_local.append(f"modelcheck_age.keras")
    else:       
       model_vect_local.append(f"model_age.keras") 

    if (isBestModelGender == True):      
       model_vect_local.append(f"modelcheck_gender.keras")
    else:       
       model_vect_local.append(f"model_gender.keras")
    
    return model_vect_local


model_vect = []

isBestModelEmotion = True
isBestModelAge = True
isBestModelGender = True

# Gets the correct model (from the "best epoch" or the "last save") to load 
# from the "emotion", the "age" and the "gender" models.
model_vect = getCorrectModelsToLoad(isBestModelEmotion, isBestModelAge, isBestModelGender)

face_classifier = cv2.CascadeClassifier(f"haarcascade_frontalface_default.xml")

classifier_emotion = load_model(f"{model_vect[0]}")
classifier_age = load_model(f"{model_vect[1]}")
classifier_gender = load_model(f"{model_vect[2]}")

# Creates the "correct" labels for the "emotion", the "age" and the "gender" model.
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

gender_labels = ['Female', 'Male']

age_labels = []
for age_val in range(0, 90, 20):
   age_labels.append(f"{age_val} - {(age_val + 19)} years old")
age_labels.append(f"80 or more years old")


cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:

        # Draw rectangle around face.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

        # Get the gray ROI for emotion detection (grayscale).
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

        # Get the color ROI for age and gender detection (color).
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (48, 48), interpolation = cv2.INTER_AREA)

        # Emotion detection (using grayscale).
        if np.sum([roi_gray]) != 0:

            # Gray.
            roi_g = roi_gray.astype('float') / 255.0
            roi_g = img_to_array(roi_g)
            roi_g = np.expand_dims(roi_g, axis = 0)

            # Emotion prediction (from gray image).
            prediction_emotion = classifier_emotion.predict(roi_g)[0]
            print(prediction_emotion)

            label_emotion = emotion_labels[prediction_emotion.argmax()]

            label_emotion_print = f"{label_emotion}."

            print(prediction_emotion.argmax())
            print(label_emotion)

            label_pos_emotion = (x, y + 40)

            cv2.putText(frame, label_emotion, label_pos_emotion, cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)

        # Age and Gender detection (using color).
        if np.sum([roi_color]) != 0:

            # Color.
            roi_c = roi_color.astype('float') / 255.0
            roi_c = img_to_array(roi_c)
            roi_c = np.expand_dims(roi_c, axis = 0)

            # Age prediction (from color image).
            prediction_age = classifier_age.predict(roi_c)[0]
            print(prediction_age)

            print(prediction_age.argmax())

            label_age = age_labels[prediction_age.argmax()]
            print(label_age)

            # Gender prediction (from color image).
            prediction_gender = classifier_gender.predict(roi_c)[0]
            print(prediction_gender)

            print(prediction_gender.argmax())
           
            label_gender = gender_labels[prediction_gender.argmax()]
            print(label_gender)

            label_gender_print = f"{label_age}."

            label_age_print = f"{label_gender}."

            label_pos_gender = (x, y + 80)

            label_pos_age = (x, y + 120)

            cv2.putText(frame, label_gender_print, label_pos_gender, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, label_age_print, label_pos_age, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:

            cv2.putText(frame,'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Age, Gender, and Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
