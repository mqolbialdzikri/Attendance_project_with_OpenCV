import cv2, os, numpy as np
from PIL import Image

wajahDir = 'datawajah'
latihDir = 'latihwajah'
def getImagesLabel (path):
    imagePaths = [os.path.join (path,f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths :
        PILImg = Image.open(imagePath).convert('L') #convert kedalam grey
        imgNum = np.array(PILImg, 'uint8' )
        id = os.path.split(imagePath)[-1].split(".")[2]
        faceID = int(id)
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces :
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
    return faceSamples,faceIDs

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("mesin sedang melakukan training data wajah. Tunggu bentar ya")
faces, IDs = getImagesLabel(wajahDir)
faceRecognizer.train(faces, np.array(IDs))

#simpan
faceRecognizer.write(latihDir +'/training.xml')
print('sebanyak {0} data wajah telah ditrainingkan ke mesin', format(len(np.unique(IDs))))
