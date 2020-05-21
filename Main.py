from model import DeepFace
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tabulate import tabulate
from colorama import Fore, Back, Style 
import cv2

warnings.filterwarnings("ignore")


Tk().withdraw()
filename = askopenfilename()
#hall we filename = 'tests/dataset/img1.jpg'
img = cv2.imread(filename)
result = DeepFace.analyze(filename)

result_dict = json.loads(result)
cv2.imshow(result_dict['dominant_emotion'],img)
cv2.waitKey(0)

dic = pd.DataFrame(result_dict["emotion"],index=['emotion'])

dic.plot.bar()

plt.show()

print("The accurecy of Each Emotion")
print(tabulate(result_dict['emotion'].items(), headers=['Emotion', 'Accurecy']))

print("The Dominant Emotion is : "+Fore.GREEN +Style.DIM+result_dict['dominant_emotion'])
cv2.destroyAllWindows()