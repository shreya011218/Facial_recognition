#LIVE TRACKING BY COMPARING TO DATABASE WHEN BUTTON IS CLICKED#
#captures pixels when a button on the screen is clicked and compares the face to the preloaded database#


import face_recognition
import cv2
import numpy as np
import time
import os
import pickle  # Library used to save/load python objects to files

# --- CONFIGURATION PATHS ---
folder_path = r"D:\shreya school\COLLAGE\z Internship\A3i\project\training photos"
db_cache_file = "face_database.pkl"

# =========================================================================
# ⭐ STEP 1: LOAD OR CREATE THE INSTANT STARTUP ENCODING DATABASE ⭐
# =========================================================================
known_face_encodings = []
known_face_names = []

# Check if the cache database file already exists on your computer
if os.path.exists(db_cache_file):
    print("Found cached database! Loading profiles instantly...")
    with open(db_cache_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print("✅ Database loaded instantly. Zero startup delay!")
else:
    print("Cache file not found. Processing training photos (One-time generation)...")
    
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
    known_face_names = ["Priyanka", "Alia", "Shreya"]
    
    # Save the processed math matrices to disk so we never have to run this again
    with open(db_cache_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("✅ Encodings generated and saved to face_database.pkl for future fast boots!")


# =========================================================================
# ⭐ STEP 2: ON-SCREEN BUTTON SETUP & INTERACTIVE UI LOGIC ⭐
# =========================================================================
window_name = 'Interactive Face Recognition'
# Bounding coordinates for our virtual button box (Top-Left Corner)
btn_x1, btn_y1 = 20, 20
btn_x2, btn_y2 = 200, 70

trigger_scan = False  # Track whether the user has clicked our scanner button

def mouse_click_handler(event, x, y, flags, param):
    """Listens to where your cursor clicks on the window."""
    global trigger_scan
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse click happened INSIDE our button coordinates
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            trigger_scan = True
            print("\nButton Clicked! Initializing face scan...")


# --- STEP 3: LIVE CAMERA INTERFACE LOOP ---
print("\nOpening camera... Click the on-screen 'SCAN FACE' button or press 'q' to quit.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Initialize the UI window and bind the mouse click handler function to it
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click_handler)

# Memory placeholders to cache coordinates so boxes stay visible between clicks
face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 🖱️ MANUAL BUTTON TRIGGER: Only executes if trigger_scan was flipped to True by your mouse
    if trigger_scan:
        # Convert frame from OpenCV BGR to face_recognition RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Locate faces on screen
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Extract facial features instantly (num_jitters=1 keeps performance rapid)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model='large')
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            label = "Unrecognized"
            
            if True in matches:
                first_match_index = matches.index(True)
                label = known_face_names[first_match_index]
            
            face_names.append(label)
            print(f"[{time.strftime('%H:%M:%S')}] Manual Scan Result: {label}")

        # Reset flag so it waits for your next click
        trigger_scan = False

    # --- DRAW THE INTERACTIVE BUTTON ---
    # Draw solid background rectangle for button
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (230, 100, 10), -1) 
    # Label the button text clear and crisp
    cv2.putText(frame, "SCAN FACE", (btn_x1 + 25, btn_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- DRAW DETECTION BOXES (Persistent from last scan data) ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 160, 0) if name != "Unrecognized" else (0, 0, 170)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame inside our active interactive window
    cv2.imshow(window_name, frame)
    
    # Alternate physical keyboard fallback: Tap spacebar to trigger scan or 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        trigger_scan = True
    elif key == ord('q'):
        print("Stopping Live Tracking.")
        break

cap.release()
cv2.destroyAllWindows()