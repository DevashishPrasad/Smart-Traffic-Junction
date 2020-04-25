# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import numpy as np

# load the face embeddings
print("[INFO] loading Road embeddings...")
data = pickle.loads(open("embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print(np.array(data["embeddings"]).shape)
# train the model used to accept the 2622-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
clf = SVC(C=5, kernel="linear", probability=True)
#clf = SVC(C=1, kernel="rbf" ,degree=3 , probability=True)
clf.fit(data["embeddings"], labels)
print(clf.score(data["embeddings"], labels))

# write the actual face recognition model to disk
f = open("classifier.pickle", "wb")
f.write(pickle.dumps(clf))
f.close()

# write the label encoder to disk
f = open("le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
