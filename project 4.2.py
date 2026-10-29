#LIVE TRACKING,PHOTO CAPTURE REGISTRATION,FILE UPLOAD REGISTRATION#


import face_recognition
import cv2
import numpy as np
import time
import os
import pickle
import shutil

# --- CONFIGURATION PATHS ---
folder_path = r"D:\shreya school\COLLAGE\z Internship\A3i\project\training photos"
db_cache_file = "face_database.pkl"


known_face_encodings = []
known_face_names = []

if os.path.exists(db_cache_file):
    print("Found cached database! Loading profiles instantly...")
    with open(db_cache_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Database loaded. Current users: {known_face_names}")
else:
    print("Cache file not found. Processing base training photos...")
    priyanka_image = face_recognition.load_image_file(f"{folder_path}\\pc1.jpg")
    alia_image     = face_recognition.load_image_file(f"{folder_path}\\alia1.jpg")
    shreya_image   = face_recognition.load_image_file(f"{folder_path}\\me1.jpg")

    known_face_encodings = [
        face_recognition.face_encodings(priyanka_image, num_jitters=50, model='large')[0],
        face_recognition.face_encodings(alia_image, num_jitters=50, model='large')[0],
        face_recognition.face_encodings(shreya_image, num_jitters=50, model='large')[0]
    ]
    known_face_names = ["Priyanka", "Alia", "Shreya"]
    
    with open(db_cache_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("Base database generated and cached!")



# STEP 2: ON-SCREEN BUTTON SETUP & INTERACTIVE UI LOGIC
window_name = 'Interactive Face Recognition'
btn_x1, btn_y1 = 20, 20
btn_x2, btn_y2 = 200, 70
trigger_scan = False  

def mouse_click_handler(event, x, y, flags, param):
    global trigger_scan
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            trigger_scan = True
            print("\nButton Clicked! Initializing face scan...")


# --- STEP 3: LIVE CAMERA INTERFACE LOOP ---
print("\n=== SYSTEM HOTKEYS ===")
print("-> Click 'SCAN FACE' button (or press SPACEBAR) to check a face.")
print("-> Press 'a' key to use 'ADD LIVE FACE' from webcam.")
print("-> Press 'f' key to use 'ADD FACE FROM FILE' (Local image upload).")
print("-> Press 'd' key to use 'DELETE FACE' from database.")
print("-> Press 'q' key to quit the program safely.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click_handler)

face_locations = []
face_names = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    #  MANUAL SCAN BUTTON TRIGGER
    if trigger_scan:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model='large')
        
        face_names = []
        for face_encoding in face_encodings:
            label = "Unrecognized"
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < 0.5:
                    label = known_face_names[best_match_index]
            
            face_names.append(label)
            print(f"[{time.strftime('%H:%M:%S')}] Scan Result: {label}")
        trigger_scan = False

    # --- UI DRAWING ELEMENTS ---
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (230, 100, 10), -1) 
    cv2.putText(frame, "SCAN FACE", (btn_x1 + 25, btn_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Updated text layout overlay to show 'd' key functionality
    cv2.putText(frame, "Keys: 'a'=Add Live | 'f'=Upload File | 'd'=Delete Face", (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 160, 0) if name != "Unrecognized" else (0, 0, 170)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # COMMAND 1: ADD LIVE FACE
    if key == ord('a'):
        print("\n[COMMAND] 'Add Live Face' initiated. Analyzing frame...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locs = face_recognition.face_locations(rgb_frame)
        
        if len(temp_locs) == 0:
            print("Error: No face detected in frame.")
        elif len(temp_locs) > 1:
            print("Error: Multiple faces detected. Keep only ONE person in frame.")
        else:
            check_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=1, model='large')[0]
            is_duplicate = False
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    print(f"Aborted: Face already matches an existing profile: '{known_face_names[best_match_index]}'.")
                    is_duplicate = True
            
            if not is_duplicate:
                print("Face verified as unique entry.")
                confirm = input("Would you like to add this live face? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    new_name = input("Enter the name for this new face: ").strip()
                    if new_name:
                        print(f"Processing matrix for '{new_name}' (Jitters=50)...")
                        new_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=50, model='large')[0]
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(new_name)
                        
                        cv2.imwrite(os.path.join(folder_path, f"{new_name.lower().replace(' ', '_')}1.jpg"), frame)
                        with open(db_cache_file, "wb") as f:
                            pickle.dump((known_face_encodings, known_face_names), f)
                        print(f"Success! Added live face for '{new_name}'.")
                    else:
                        print("Action canceled. Name cannot be blank.")

    # 🔵 COMMAND 2: ADD FACE FROM LOCAL FILE
    elif key == ord('f'):
        print("\n[COMMAND] 'Add Face From File' initiated.")
        print("Look at your VS Code Terminal output below!")
        
        file_path = input("Enter or drag-and-drop the full path to the image file: ").strip().strip("'\"")
        
        if not os.path.exists(file_path):
            print("Error: File path does not exist.")
        else:
            try:
                uploaded_img = face_recognition.load_image_file(file_path)
                temp_locs = face_recognition.face_locations(uploaded_img)
                
                if len(temp_locs) == 0:
                    print("Error: Could not detect any clear faces in that file.")
                elif len(temp_locs) > 1:
                    print("Error: Multiple faces found. Please isolate a single face image.")
                else:
                    check_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=1, model='large')[0]
                    is_duplicate = False
                    
                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.5:
                            print(f"Aborted: This file already matches an existing profile: '{known_face_names[best_match_index]}'.")
                            is_duplicate = True
                    
                    if not is_duplicate:
                        print("Image file verified as a unique record.")
                        new_name = input("Enter the identity/name for this image file: ").strip()
                        
                        if new_name:
                            print(f"Generating high-quality signature matrix for '{new_name}'...")
                            new_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=50, model='large')[0]
                            
                            known_face_encodings.append(new_encoding)
                            known_face_names.append(new_name)
                            
                            extension = os.path.splitext(file_path)[1]
                            new_filename = f"{new_name.lower().replace(' ', '_')}1{extension}"
                            shutil.copy(file_path, os.path.join(folder_path, new_filename))
                            
                            with open(db_cache_file, "wb") as f:
                                pickle.dump((known_face_encodings, known_face_names), f)
                                
                            print(f"Success! File database expanded. '{new_name}' registered successfully.")
                        else:
                            print("Action canceled. Name cannot be blank.")
            except Exception as e:
                print(f"An error occurred while parsing the file: {e}")

    #  NEW COMMAND 3: DELETE FACE PROFILE ('d' key)
    elif key == ord('d'):
        print("\n[COMMAND] 'Delete Face' profile workflow activated.")
        print("Look at your VS Code Terminal output below!")
        
        if not known_face_names:
            print("The face database is completely empty. Nothing to delete!")
        else:
            print(f"Current Profiles in Database: {known_face_names}")
            name_to_delete = input("Enter the exact name profile you wish to delete: ").strip()
            
            # Case-insensitive matching lookup loop
            matched_indices = [i for i, registered_name in enumerate(known_face_names) if registered_name.lower() == name_to_delete.lower()]
            
            if not matched_indices:
                print(f"Error: '{name_to_delete}' does not match any profile in our database records.")
            else:
                target_index = matched_indices[0]
                target_name_exact = known_face_names[target_index]
                
                # Double-check confirmation step to ensure safety
                confirm = input(f"Are you sure you want to permanently delete '{target_name_exact}'? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    # 1. Pop out data entries from current runtime system arrays
                    del known_face_encodings[target_index]
                    del known_face_names[target_index]
                    
                    # 2. Overwrite the file system cache file database
                    with open(db_cache_file, "wb") as f:
                        pickle.dump((known_face_encodings, known_face_names), f)
                    print(f"System memory updated. Profile '{target_name_exact}' removed.")
                    
                    # 3. Automatic storage cleanup: Find and destroy their photo file
                    base_img_format = target_name_exact.lower().replace(' ', '_') + "1"
                    file_purged = False
                    for ext in ['.jpg', '.jpeg', '.png']:
                        potential_file_path = os.path.join(folder_path, base_img_format + ext)
                        if os.path.exists(potential_file_path):
                            os.remove(potential_file_path)
                            file_purged = True
                    
                    if file_purged:
                        print(f"Cleaned up associated file '{base_img_format}' from training photos folder.")
                    else:
                        print("Note: No matching training photo found to remove, profile was database cache only.")
                else:
                    print("Deletion process canceled safely.")

    elif key == ord(' '):
        trigger_scan = True
    elif key == ord('q'):
        print("Stopping Live Tracking.")
        break

cap.release()
cv2.destroyAllWindows()