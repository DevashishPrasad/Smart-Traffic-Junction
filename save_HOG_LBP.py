import cv2
import numpy as np
from imutils import paths
from skimage.feature import hog,local_binary_pattern
import os
import pickle


imagePaths = list(paths.list_images("dataset"))

Extracted_Names = []
Extracted_Embeddings = []

eps=1e-7
numPoints = 24
radius = 8

for (i, imagePath) in enumerate(imagePaths):
	if(int(i) >= 10000 and int(i) <= 15000):
		continue
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	img = cv2.imread(imagePath)
	# compute the Local Binary Pattern representation
	# of the image, and then use the LBP representation
	# to build the histogram of patterns
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hist = local_binary_pattern(gray, numPoints, radius, method="uniform")
	(hist, _) = np.histogram(hist.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
	#normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
 
	lbp_embedding = hist
	print("LBP",lbp_embedding.shape)
	hog_embedding = hog(gray, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(1, 1), visualize=False, multichannel=False)
	print("HOG",hog_embedding.shape)	
	embedding = np.append(hog_embedding.ravel(),lbp_embedding.ravel())
	print("TOTAL",embedding.shape)	
	Extracted_Names.append(name)
	Extracted_Embeddings.append(embedding)

# dump the HOG embeddings + names to disk
print("[INFO] serializing encodings...")
data = {"embeddings": Extracted_Embeddings, "names": Extracted_Names}
f = open("embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
