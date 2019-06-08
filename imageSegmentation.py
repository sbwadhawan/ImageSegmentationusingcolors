from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

image=cv2.imread('sample_img.jpeg')
print("The type of this input is {}".format(type(image)))


scale_percent=50
width=int(image.shape[1]*scale_percent/100)
height=int(image.shape[0]*scale_percent/100)
dim=(width,height)

image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
cv2.imshow('result.png',image)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image=image.reshape(image.shape[0],image.shape[1]*3)
print(image.shape)


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]),int(color[1]),int(color[2]))

clf=KMeans(n_clusters=3)
labels=clf.fit_predict(image)
count=Counter(labels)#makes a dictionary
print(count)

center_colors=clf.cluster_centers_
print(center_colors)

ordered_colors=[center_colors[i] for i in count.keys()]
#print(len(ordered_colors[2]))
hex_colors = [RGB2HEX(ordered_colors[i]) for i in count.keys()]
rgb_colors = [ordered_colors[i] for i in count.keys()]

if True:
    plt.figure(figsize = (8, 6))
    plt.pie(count.values(), labels = hex_colors, colors = hex_colors)
    plt.show()



