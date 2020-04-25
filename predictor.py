# import the necessary packages
import numpy as np
import pickle
import cv2
from skimage.feature import hog,local_binary_pattern
# load the actual face recognition model along with the label encoder
print("[INFO] loading model")
recognizer = pickle.loads(open("classifier.pickle", "rb").read())
#pca = pickle.loads(open("pca.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())
print("[INFO] Model loaded successfully")

cap = cv2.VideoCapture('Junction2.avi')

print("[INFO] Starting with video")
ic = 0
jc = 0
eps=1e-7
numPoints = 24
radius = 8

while True:
    _,img = cap.read()
    img = cv2.resize(img,(800,600))
    roi = img[80:435,270:670]
    col = roi.copy()
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    h,w = roi.shape
    
    ic+=1
    if(ic <= 150):
    	cv2.imshow('temp',roi)
    	cv2.waitKey(1)
    	continue
    
    if(ic%8 == 0):
    	for i in range(44,h,44):
    		for j in range(44,w,44):
    			box = roi[i-44:i,j-44:j]
    			lbp = local_binary_pattern(box, numPoints, radius, method="uniform")
    			(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    			# normalize the histogram
    			hist = hist.astype("float")
    			hist /= (hist.sum() + eps)
 
    			lbp_embedding = hist
    			hog_embedding = hog(box, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(1, 1), visualize=False, multichannel=False)
    			embedding = np.append(hog_embedding.ravel(),lbp_embedding)
    			#embedding = pca.transform(embedding.reshape(1, -1))  
    			prediction = recognizer.predict(embedding.reshape(1, -1))
    			#cv2.rectangle(nroi,(j,i),(j-121,i-121),(255,0,0),2)
    			if(prediction == 1):
	    			cv2.rectangle(col,(j,i),(j-39,i-39),(0,0,255),1)
	    		else:
		    		cv2.rectangle(col,(j,i),(j-39,i-39),(0,255,0),1)
    			jc+=1
    	cv2.imshow('temp2',col)
    	cv2.waitKey(1)
cv2.destroyAllWindows()
