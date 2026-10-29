#FACIAL IDENTIFICATION AND RECOGNITION#
#both test and training photos are uploaded as files and then compared to identify the person#


import face_recognition
import cv2
import numpy as np

# 1. Load Training Images (No longer using D:\\ paths, just the filenames)
priyanka_image = face_recognition.load_image_file("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\pc1.jpg")
alia_image = face_recognition.load_image_file("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\alia1.jpg")

# 2. Generate Face Encodings
priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]
alia_encoding = face_recognition.face_encodings(alia_image)[0]

known_face_encodings = [
    priyanka_encoding,  
    alia_encoding
]

known_face_names = [
    "priyanka",
    "alia"
]

# 3. Load the Test Image using OpenCV
image = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\alia2.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. Detect Faces and Generate Encodings
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# 5. Draw Bounding Boxes and Labels
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw the green rectangle around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Place text label above the rectangle
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow('Face Recognition Result', image)

# 2. Prevent the desktop window from instantly closing itself
cv2.waitKey(0)
cv2.destroyAllWindows()
