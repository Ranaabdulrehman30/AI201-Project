import numpy
import cv2
import face_recognition

imgRana = face_recognition.load_image_file('images/rana.JPG')
imgRana = cv2.cvtColor(imgRana, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('images/ranaTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

face_location = face_recognition.face_locations(imgRana)[0]
en_Code = face_recognition.face_encodings(imgRana)[0]

print(imgRana)
cv2.rectangle(imgRana, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 255), 2)


cv2.imshow("rana", imgRana)
cv2.imshow("test", imgTest)
cv2.waitKey(0)