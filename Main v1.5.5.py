import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

files = os.listdir(r"C:\Work\Python_Project\images")

known_face_encodings = []

for file in files:
    image = face_recognition.load_image_file(f"images\{file}")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)

path = "C:\Work\Python_Project\images"
known_face_names = [os.path.splitext(filename)[0] for filename in os.listdir(path)]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 12), font, 0.50, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()