# run myenv\Scripts\activate everytime u open 
# pip install face_recognition opencv-python matplotlib


import face_recognition
import cv2
import numpy as np

# 1. Load training photos using OpenCV
priyanka_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\pc1.jpg")
alia_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\alia1.jpg")
shreya_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\me1.jpg")

# Convert from BGR (OpenCV standard) to RGB (face_recognition standard)
priyanka_image = cv2.cvtColor(priyanka_bgr, cv2.COLOR_BGR2RGB)
alia_image = cv2.cvtColor(alia_bgr, cv2.COLOR_BGR2RGB)
shreya_image = cv2.cvtColor(shreya_bgr, cv2.COLOR_BGR2RGB)

# 2. Extract facial encodings
priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]
alia_encoding = face_recognition.face_encodings(alia_image)[0]
shreya_encoding = face_recognition.face_encodings(shreya_image)[0]

known_face_encodings = [
    priyanka_encoding,
    alia_encoding,
    shreya_encoding
]

known_face_names = [
    "priyanka",
    "alia",
    "shreya"
]

# 3. Load the test image
# PRO TIP: Right now you are testing with "me1.jpg", which is the exact same photo 
# used for training. It will definitely work! But for a true test of the AI, 
# try taking a brand new picture of yourself (like "me2.jpg") and use that here instead!
test_image = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\me2.jpg")
rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# 4. Find face locations and encodings in the test image
face_locations = face_recognition.face_locations(rgb_test_image)
face_encodings = face_recognition.face_encodings(rgb_test_image, face_locations)

# 5. Loop through found faces and draw bounding boxes
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    print(f"Found face: {name}")

    # Draw the green bounding box rectangle
    cv2.rectangle(
        test_image,
        (left, top),
        (right, bottom),
        (0, 255, 0),
        2
    )

    # Draw the text label string above the box
    cv2.putText(
        test_image,
        name,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# 6. Windows Desktop Image Display (Cleaned up the duplicates!)
cv2.imshow('Face Recognition Result', test_image)

# Keeps the window open until you click it and press ANY key on your keyboard
cv2.waitKey(0)
cv2.destroyAllWindows()

