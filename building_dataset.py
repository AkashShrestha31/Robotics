import cv2
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import matplotlib.pyplot as plt
haar_face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
def detect_Face(image,i):
		# image=cv2.imread(path)
		image2=image.copy()
		gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5);  #1.3
		j=0
		#print the number of faces found 
		print('Faces found: ', len(faces)) 
		for (x, y, w, h) in faces:     
		         cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 4)
		         try:
		         	plt.imsave("./captureim/change_your_name"+str(j)+str(i)+".jpg",cv2.cvtColor(image2[y-10:y+10+h,x-10:x+w+10],cv2.COLOR_BGR2RGB))
		         	j=j+1
		         except e:
		         	print("error message",e)
		         	continue
def capture():
	i=0
	try:
		os.mkdir('./captureim')
	except:
		pass
	data=cv2.VideoCapture(0)
	ret=True 
	while ret:
		ret,img=data.read()
		# cv2.imwrite("./captureim/capture"+str(count)+".jpg",img)
		detect_Face(img,i)
		i=i+1
		cv2.imshow("afafa",img)

		if cv2.waitKey(30) & 0xFF== ord('q'):
			return
	cv2.destroyAllWindows()
	cv2.VideoCapture(0).release()
capture()
