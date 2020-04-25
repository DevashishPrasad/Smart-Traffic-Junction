import cv2
import numpy as np

cap = cv2.VideoCapture('Junction2.avi')

ic = 0
jc = 0
kc = 0
while True:
    _,img = cap.read()
    img = cv2.resize(img,(800,600))
    roi = img[80:435,270:670]
    h,w,_ = roi.shape
    ic+=1
    
    if(kc == 201):
    	break
    
    if(ic%15 == 0):
    	for i in range(44,h,34):
    		for j in range(44,w,34):
    			nroi = roi.copy()
    			box = roi[i-44:i,j-44:j]
    			#cv2.rectangle(roi,(j,i),(j-44,i-44),(255,0,0),2)
    			print("Saved image " + str(jc))
    			cv2.imwrite('dataset/'+str(jc)+'.jpg',box)
    			jc+=1
    			
    		#cv2.imshow("op",roi)
    		#cv2.waitKey(0)			
    	kc += 1		
    print("------------------------------Taken---------------------------------->",kc)			    
   
    cv2.imwrite("op.jpg",roi)
cv2.destroyAllWindows()
