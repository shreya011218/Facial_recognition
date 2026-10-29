import face_recognition
import cv2
import numpy as np

# training photos using OpenCV
priyanka_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\pc1.jpg")
alia_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\alia1.jpg")
shreya_bgr = cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\me1.jpg")


# testing photo (working)
test_image=cv2.imread("D:\\shreya school\\COLLAGE\\z Internship\\A3i\\project\\training photos\\pc2.jpg")





cv2.imshow('Facial Recognition Result', test_image)

cv2.waitKey(20)
cv2.destroyAllWindows()
