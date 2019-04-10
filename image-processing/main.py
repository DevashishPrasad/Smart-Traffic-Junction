from __future__ import division
import numpy as np
import cv2
import imutils

# font on the frame
font = cv2.FONT_HERSHEY_SIMPLEX

# ROI extraction range
ys = 140
ye = 480
xs = 160
xe = 480

# calculating distance map
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

# Reading the video
cap = cv2.VideoCapture("video.mp4")

# frame 1
_, frame1 = cap.read()
frame1 = cv2.resize(frame1, (640,480))
frame1 = frame1[ys:ye, xs:xe]

# frame 2
_, frame2 = cap.read()
frame2 = cv2.resize(frame2, (640,480))
frame2 = frame2[ys:ye, xs:xe]

while(True):
    ret, frame3 = cap.read()
    frame3 = cv2.resize(frame3, (640,480))

    frame3 = frame3[ys:ye, xs:xe]
    frame4 = frame3.copy()

    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    blur = cv2.GaussianBlur(dist, (11,11), 0)

    # apply thresholding
    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    # cnts = imutils.grab_contours(cnts)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
	# loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < 200:
            cv2.drawContours(mask, [c], -1, 0, -1)
            continue

    # remove the contours from the image and show the resulting images
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    temp = cv2.imread("temp.jpg")
    temp = cv2.resize(temp, (xe - xs,ye - ys))
    # Bitwise-AND mask and original image
    temp = cv2.bitwise_and(temp,temp, mask= thresh)

    weighted = cv2.addWeighted(frame4, 0.6, temp, 0.4, 0)

    load = round((cv2.countNonZero(thresh) / ((480 - 140) * (480 -160))), 4) * 100

    # show the output
    cv2.imshow('weight', weighted)
    cv2.imshow('dist', thresh)
    cv2.imshow('blur', blur)
    cv2.putText(frame4, "LOAD = " + str(load) + "%", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) 
    cv2.imshow('frame', frame4)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
