import cv2, os, numpy as np
import datetime

wajahDir = 'datawajah'
latihDir = 'latihwajah'
rekapdataDir = 'rekapdataabsen'

cam = cv2.VideoCapture(0)
cam.set(3,640) #untuk mengubah lebar cam
cam.set(4, 480) #untuk mengubah tinggi cam

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['tidak dikenali','Sukry','M Qolbi Al Zikri']

def savetocsvfile (ids, waktus) :
    dt=datetime.date.today()
    
    with open (str (dt) +'_recap.csv', '+w') as f:
        for i in range (len(ids)) :
            id = ids [i]
            waktu = waktus [i]
            temp = str(waktu) +',' + names[id]
            f.writelines(temp)
            print (temp)

minWidth = 0.1*cam.get (3)
minHeight = 0.1*cam.get(4)
tercatatID = []
tercatatWaktu = []

print ("presensi dimulai")
while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu,1.2 ,5,minSize=(round(minWidth), round(minHeight)))
    # print ('faces',faces)
    for (x, y, w, h) in faces :
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (225,0,220), 2 )
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w] )
        #print ('confidence',confidence)
        if confidence<=50 :
            nameID = names [id]
            confidenceTxt = "{0}%".format(round(100-confidence))
            if(not(id in tercatatID)):
                dt=datetime.datetime.now()
                print(dt, nameID, "sudah tercatat")
                tercatatID.append(id)
                tercatatWaktu.append(dt)
        else:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100-confidence))
        cv2.putText(frame,str(nameID),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText (frame,str(confidenceTxt),(x+5,y+h-5),font,1,(255,255,0),1)

    cv2.imshow('Pengenalan absen cuy', frame)
    if cv2.waitKey(1) & 0xFF == ord ('Z') or len(tercatatID)==len(names)-1:
        break

print ("Presensi selesai \n")
print('tercatat', tercatatID)
savetocsvfile (tercatatID,tercatatWaktu)

cam.release()
cv2.destroyAllWindows()