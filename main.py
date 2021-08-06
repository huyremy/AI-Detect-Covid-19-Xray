import tensorflow 
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
virus = load_model('virus.h5')
def output(model,img_path,size):
    img_path=img_path
    img = image.load_img(img_path,target_size=size)
    img_arr = image.img_to_array(img,dtype='double')
    img_arr=img_arr/255
    img_arr=np.expand_dims(img_arr,axis=0)
    print(img_arr)
    result=model.predict_classes(img_arr)
    if result==1:
        print("Dương tính")
    elif result==0:
        print("Âm tính")
img = mpimg.imread('test.jpeg')
plt.imshow(img)
output(virus,'test.jpeg',(224,224))
