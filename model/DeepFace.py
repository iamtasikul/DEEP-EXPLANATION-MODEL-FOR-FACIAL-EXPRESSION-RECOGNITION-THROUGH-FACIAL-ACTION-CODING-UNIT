from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

#from basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
#from extendedmodels import Age, Gender, Race, Emotion
#from commons import functions, distance as dst

from model import  Emotion
from deepface.commons import functions, distance as dst


def analyze(img_path):
	
	if type(img_path) == list:
		img_paths = img_path.copy()
		bulkProcess = True
	else:
		img_paths = [img_path]
		bulkProcess = False
	
	#---------------------------------
	

	emotion_model = Emotion.loadModel()
	
	resp_objects = []
	for img_path in img_paths:
		
		if type(img_path) != str:
			raise ValueError("You should pass string data type for image paths but you passed ", type(img_path))
		
		if os.path.isfile(img_path) != True:
			raise ValueError("Confirm that ",img_path," exists")
		
		resp_obj = "{"
		
		
		emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
		img = functions.detectFace(img_path, (48, 48), True)
				
		emotion_predictions = emotion_model.predict(img)[0,:]
				
		sum_of_predictions = emotion_predictions.sum()
				
		emotion_obj = "\"emotion\": {"
		for i in range(0, len(emotion_labels)):
			emotion_label = emotion_labels[i]
			emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
					
			if i > 0: emotion_obj += ", "
					
			emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)
				
		emotion_obj += "}"
				
		emotion_obj += ", \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])
				
		resp_obj += emotion_obj
		resp_obj += "}"
	return resp_obj
	
	if bulkProcess == True:
		return resp_objects

