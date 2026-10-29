#TRACKING AND RECOGNITION FROM DATABASE#
#test photo is captured via webcam and compared with test photos#


import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt


def take_photo(window_title='Webcam Capture'):
    """Opens a native desktop webcam window. Press 'SPACE' to capture a photo, or 'q' to quit."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure it isn't being used by Zoom/Teams/Meet.")

    print(f"\n--- {window_title} ---")
    print("Instructions: Press SPACEBAR to capture the photo, or 'q' to quit.")
    
    captured_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Displays the live webcam stream on screen
        cv2.imshow(window_title, frame)
        
        key = cv2.waitKey(1) & 0xFF
        # Press Spacebar to capture
        if key == ord(' '):
            captured_img = frame.copy()
            print("Photo captured successfully!")
            break
        # Press 'q' to quit early
        elif key == ord('q'):
            print("Capture canceled.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return captured_img

print("Loading training photos from directory...")
folder_path = r"D:\shreya school\COLLAGE\z Internship\A3i\project\training photos"

# Load the three distinct training images using face_recognition
priyanka_image = face_recognition.load_image_file(f"{folder_path}\\pc1.jpg")
alia_image     = face_recognition.load_image_file(f"{folder_path}\\alia1.jpg")
shreya_image   = face_recognition.load_image_file(f"{folder_path}\\me1.jpg")

# SPEED OPTIMIZATION: Changed num_jitters from 50 to 1 for instant script startup
known_face_encodings = [
    face_recognition.face_encodings(priyanka_image, num_jitters=50)[0],
    face_recognition.face_encodings(alia_image, num_jitters=50)[0],
    face_recognition.face_encodings(shreya_image, num_jitters=50)[0]
]

# Create a corresponding list of names matching the order of encodings above
known_face_names = ["Priyanka", "Alia", "Shreya"]

print("Training images loaded and encoded successfully!")


# Capture TEST Face Live 
print("\n Opening camera for TEST face...")
test_frame = take_photo('Capture TEST Face (Press SPACE)')

if test_frame is None:
    print("No test face captured. Exiting.")
    exit()

cv2.imwrite('test.jpg', test_frame)

# Face recognition needs an RGB version to process locations and encodings
rgb_test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_test_frame)

# SPEED OPTIMIZATION: Changed num_jitters from 23 to 1 and omitted 'large' model 
# to make the webcam image process and pop up instantly.
face_encodings = face_recognition.face_encodings(
    rgb_test_frame,
    face_locations,
    num_jitters=50
)


for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Comparing the captured live face encoding against all 3 loaded profiles
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
    # Default assumptions if no matches hit
    label = "Unrecognized"
    color = (0, 0, 170) # Red for stranger

    # Checking if database match exists
    if True in matches:
        first_match_index = matches.index(True)
        label = known_face_names[first_match_index]  # Pull exact string label
        color = (0, 160, 0) # Green for match

    # Draw on the original test_frame (BGR format) for display later
    cv2.rectangle(test_frame, (left, top), (right, bottom), color, 2)
    cv2.putText(test_frame, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    print(f"Result: {label}")

# Display final static result plot using matplotlib on your desktop
plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
