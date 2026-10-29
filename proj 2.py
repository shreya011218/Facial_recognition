# run myenv\Scripts\activate everytime u open 
# pip install face_recognition opencv-python matplotlib

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
            
        # Display the live webcam stream to the user
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

# --- STEP 1: Capture KNOWN Face Live ---
print("Opening camera for KNOWN face...")
known_frame = take_photo('Capture KNOWN Face (Press SPACE)')

if known_frame is None:
    print("No known face captured. Exiting.")
    exit()

# Save it to disk just like your original code did
cv2.imwrite('known.jpg', known_frame)

# OpenCV loads BGR, so convert to RGB for face_recognition
known_image_rgb = cv2.cvtColor(known_frame, cv2.COLOR_BGR2RGB)

# Extract facial signature matrix
known_encoding = face_recognition.face_encodings(
    known_image_rgb,
    num_jitters=50,
    model='large'
)[0]


# --- STEP 2: Capture TEST Face Live ---
print("\nOpening camera for TEST face...")
test_frame = take_photo('Capture TEST Face (Press SPACE)')

if test_frame is None:
    print("No test face captured. Exiting.")
    exit()

cv2.imwrite('test.jpg', test_frame)

# Face recognition needs an RGB version to process locations and encodings
rgb_test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_test_frame)
face_encodings = face_recognition.face_encodings(
    rgb_test_frame,
    face_locations,
    num_jitters=23,
    model='large'
)


# --- STEP 3: Compare and Plot ---
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    match = face_recognition.compare_faces([known_encoding], face_encoding)[0]

    label = "Recognized" if match else "Unrecognized"
    color = (0, 255, 0) if match else (0, 0, 255) # Green for match, Red for stranger

    # Draw on the original test_frame (BGR format) for display later
    cv2.rectangle(test_frame, (left, top), (right, bottom), color, 2)
    cv2.putText(test_frame, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    print(f"Result: {label}")

# Display final static result plot using matplotlib on your desktop
plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()