#LIVE TRACKING AND RECOGNITION FROM UPLOADED FILES#
# tracks and captures frame every one second and shows who is in the frame 


import face_recognition
import cv2
import numpy as np
import time

print("Loading training photos from directory...")
folder_path = r"D:\shreya school\COLLAGE\z Internship\A3i\project\training photos"

# Loading the training images using face_recognition
priyanka_image = face_recognition.load_image_file(f"{folder_path}\\pc1.jpg")
alia_image     = face_recognition.load_image_file(f"{folder_path}\\alia1.jpg")
shreya_image   = face_recognition.load_image_file(f"{folder_path}\\me1.jpg")

# Extracting face matrices using high-quality settings
known_face_encodings = [
    face_recognition.face_encodings(priyanka_image, num_jitters=50, model='large')[0],
    face_recognition.face_encodings(alia_image, num_jitters=50, model='large')[0],
    face_recognition.face_encodings(shreya_image, num_jitters=50, model='large')[0]
]

# Create a corresponding list of names
known_face_names = ["Priyanka", "Alia", "Shreya"]
print("Training images loaded and encoded successfully!")


# --- STEP 2: LIVE TRACKING & AUTOMATIC SCANNING LOOP ---
print("\nOpening camera for Live Tracking... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Variables to hold tracking state and timing
last_processed_time = 0.0  # Tracks when the AI last scanned a face
face_locations = []        # Stores the current box coordinates
face_names = []            # Stores the names of found faces

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()

    # AUTOMATIC TIMER check: Has 1.0 second passed since the last scan?
    if current_time - last_processed_time >= 1.0:
        # Convert frame from OpenCV BGR to face_recognition RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Locate faces on screen
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Extract facial features (num_jitters set to 1 for fast real-time speed)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model='large')
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            label = "Unrecognized"
            
            if True in matches:
                first_match_index = matches.index(True)
                label = known_face_names[first_match_index]
            
            face_names.append(label)
            print(f"[{time.strftime('%H:%M:%S')}] Auto-Scanned Result: {label}")

        # Update our timer anchor point to the current timestamp
        last_processed_time = current_time

    # --- STEP 3: DRAW BOXES (Using the latest cached analysis data) ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Choose color based on verification state
        color = (0, 160, 0) if name != "Unrecognized" else (0, 0, 170) # Green for match, Red for stranger
        
        # Draw the rectangle bounding boxes and text labels
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the continuous live stream window
    cv2.imshow('Live Automatic Face Recognition', frame)
    
    # Break the loop safely if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping Live Tracking.")
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()