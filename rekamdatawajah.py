import cv2, os
wajahDir = 'datawajah'
cam = cv2.VideoCapture(0)

cam.set(3,640) #untuk mengubah lebar cam
cam.set(4, 480) #untuk mengubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceID = input("masukkan face ID yang akan direkam datanya [kemudian tekan enter]")
print("hadapkan wajah anda kedepan camera...Tunggu beberapa saat")
ambilData = 1
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale (abuAbu,1.5,5)
    for (x, y, w, h) in faces :
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (225,0,220), 2 )
        namaFile = 'wajah.' + str(faceID)+ '.'+ str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/.'+ namaFile, frame)
        ambilData += 1
    cv2.imshow('kameralaptop', frame)
    if cv2.waitKey(1) & 0xFF == ord ('Z'):
        break
    elif ambilData>15 :
        break

print ("Pengambilan data selesai")
cam.release()
cv2.destroyAllWindows()