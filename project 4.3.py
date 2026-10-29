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
history_file = "history_log.pkl"  # New file for persistent history tracking

known_face_encodings = []
known_face_names = []

if os.path.exists(db_cache_file):
    print("[INFO] Found cached database! Loading profiles instantly...")
    with open(db_cache_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"[SUCCESS] Database loaded instantly. Current users: {known_face_names}")
else:
    print("[INFO] Cache file not found. Processing base training photos...")
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
    print("[SUCCESS] Base database generated and cached!")


detection_history = []
if os.path.exists(history_file):
    print("[INFO] Found past session logs! Loading historical scan data...")
    try:
        with open(history_file, "rb") as f:
            detection_history = pickle.load(f)
        print(f"[SUCCESS] History loaded. Total past records found: {len(detection_history)}")
    except Exception as e:
        print(f"[WARNING] History file could not be read, resetting history logs: {e}")
        detection_history = []
else:
    print("[INFO] No past history file discovered. Starting a fresh log track.")


window_name = 'Interactive Face Recognition'
btn_x1, btn_y1 = 20, 20
btn_x2, btn_y2 = 200, 70
trigger_scan = False  
show_history = False     # Toggle flag for displaying history panel overlay

def mouse_click_handler(event, x, y, flags, param):
    global trigger_scan
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            trigger_scan = True
            print("\nButton Clicked! Initializing face scan...")


# --- STEP 3: LIVE CAMERA INTERFACE LOOP ---
print("\n=== SYSTEM HOTKEYS ===")
print("-> Click 'SCAN FACE' button (or press SPACEBAR) to check a face.")
print("-> Press 'h' key to TOGGLE the on-screen Detection History panel.")
print("-> Press 'a' key to 'ADD LIVE FACE' | 'f' key to 'UPLOAD FILE'.")
print("-> Press 'd' key to 'DELETE FACE' | 'q' key to quit safely.")

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
        print("[ERROR] Failed to grab frame.")
        break
        
    frame_h, frame_w, _ = frame.shape

    # MANUAL SCAN BUTTON TRIGGER
    if trigger_scan:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model='large')
        
        timestamp = time.strftime('%H:%M:%S')
        face_names = []
        
        # If scan triggered but no face is in frame, log an empty attempt
        if not face_encodings:
            detection_history.append(("No Face Detected", timestamp))
            print(f"[{timestamp}] Scan Result: No face detected.")
            
        for face_encoding in face_encodings:
            label = "Unrecognized"
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < 0.5:
                    label = known_face_names[best_match_index]
            
            face_names.append(label)
            
            # Save the scan data into memory logs
            detection_history.append((label, timestamp))
            print(f"[{timestamp}] Scan Result: {label}")
            
        # Write updated history down to local storage instantly
        with open(history_file, "wb") as f:
            pickle.dump(detection_history, f)
            
        trigger_scan = False

    # --- UI DRAWING ELEMENTS ---
    # 1. Draw "SCAN FACE" button
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (230, 100, 10), -1) 
    cv2.putText(frame, "SCAN FACE", (btn_x1 + 25, btn_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 2. Lower Control Help Bar
    cv2.putText(frame, "Keys: 'a'=Add | 'f'=File | 'd'=Del | 'h'=Toggle History Panel", (20, frame_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Target box rendering over detected coordinates
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 160, 0) if name != "Unrecognized" and name != "No Face Detected" else (0, 0, 170)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 4. DYNAMIC OVERLAY PANEL: RENDER DETECTION LOG HISTORY ON-SCREEN
    if show_history:
        panel_w = 240
        x_start = frame_w - panel_w
        
        # Create a dark semi-transparent sidebar slice overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, 0), (frame_w, frame_h), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Border outline separator line
        cv2.line(frame, (x_start, 0), (x_start, frame_h), (70, 70, 70), 1)
        
        # Header string
        cv2.putText(frame, "SCAN LOG HISTORY", (x_start + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
        cv2.line(frame, (x_start + 15, 45), (frame_w - 15, 45), (100, 100, 100), 1)
        
        # Display the 8 most recent entries (including items loaded from previous sessions)
        y_pos = 80
        recent_logs = detection_history[-8:]
        
        if not recent_logs:
            cv2.putText(frame, "No records found.", (x_start + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        else:
            for log_name, log_time in recent_logs:
                # Color code status strings inside the display board panel
                text_color = (255, 255, 255)
                if log_name == "Unrecognized":
                    text_color = (100, 100, 255)
                elif log_name == "No Face Detected":
                    text_color = (150, 150, 150)
                else:
                    text_color = (120, 255, 120)
                
                log_string = f"[{log_time}] {log_name}"
                cv2.putText(frame, log_string, (x_start + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
                y_pos += 30

    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # COMMAND: TOGGLE HISTORY PANEL DISPLAY ('h' key)
    if key == ord('h'):
        show_history = not show_history
        print(f"\n[TOGGLE] On-screen logs viewport overlay visibility set to: {show_history}")

    # COMMAND: ADD LIVE FACE
    elif key == ord('a'):
        print("\n[COMMAND] 'Add Live Face' initiated. Analyzing frame...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locs = face_recognition.face_locations(rgb_frame)
        
        if len(temp_locs) == 0:
            print("[ERROR] No face detected in frame.")
        elif len(temp_locs) > 1:
            print("[ERROR] Multiple faces detected. Keep only ONE person in frame.")
        else:
            check_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=1, model='large')[0]
            is_duplicate = False
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    print(f"[ABORTED] Face already matches an existing profile: '{known_face_names[best_match_index]}'.")
                    is_duplicate = True
            
            if not is_duplicate:
                print("[INFO] Face verified as unique entry.")
                confirm = input("Would you like to add this live face? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    new_name = input("Enter the name for this new face: ").strip()
                    if new_name:
                        print(f"[INFO] Processing matrix for '{new_name}' (Jitters=50)...")
                        new_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=50, model='large')[0]
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(new_name)
                        
                        cv2.imwrite(os.path.join(folder_path, f"{new_name.lower().replace(' ', '_')}1.jpg"), frame)
                        with open(db_cache_file, "wb") as f:
                            pickle.dump((known_face_encodings, known_face_names), f)
                        print(f"[SUCCESS] Added live face for '{new_name}'.")
                    else:
                        print("[ERROR] Action canceled. Name cannot be blank.")

    # COMMAND: ADD FACE FROM LOCAL FILE
    elif key == ord('f'):
        print("\n[COMMAND] 'Add Face From File' initiated.")
        print("[INPUT] Look at your VS Code Terminal output below!")
        
        file_path = input("Enter or drag-and-drop the full path to the image file: ").strip().strip("'\"")
        
        if not os.path.exists(file_path):
            print("[ERROR] File path does not exist.")
        else:
            try:
                uploaded_img = face_recognition.load_image_file(file_path)
                temp_locs = face_recognition.face_locations(uploaded_img)
                
                if len(temp_locs) == 0:
                    print("[ERROR] Could not detect any clear faces in that file.")
                elif len(temp_locs) > 1:
                    print("[ERROR] Multiple faces found. Please isolate a single face image.")
                else:
                    check_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=1, model='large')[0]
                    is_duplicate = False
                    
                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.5:
                            print(f"[ABORTED] This file already matches an existing profile: '{known_face_names[best_match_index]}'.")
                            is_duplicate = True
                    
                    if not is_duplicate:
                        print("[INFO] Image file verified as a unique record.")
                        new_name = input("Enter the identity/name for this image file: ").strip()
                        
                        if new_name:
                            print(f"[INFO] Generating high-quality signature matrix for '{new_name}'...")
                            new_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=50, model='large')[0]
                            
                            known_face_encodings.append(new_encoding)
                            known_face_names.append(new_name)
                            
                            extension = os.path.splitext(file_path)[1]
                            new_filename = f"{new_name.lower().replace(' ', '_')}1{extension}"
                            shutil.copy(file_path, os.path.join(folder_path, new_filename))
                            
                            with open(db_cache_file, "wb") as f:
                                pickle.dump((known_face_encodings, known_face_names), f)
                                
                            print(f"[SUCCESS] File database expanded. '{new_name}' registered successfully.")
                        else:
                            print("[ERROR] Action canceled. Name cannot be blank.")
            except Exception as e:
                print(f"[ERROR] An error occurred while parsing the file: {e}")

    # COMMAND: DELETE FACE PROFILE
    elif key == ord('d'):
        print("\n[COMMAND] 'Delete Face' profile workflow activated.")
        print("[INPUT] Look at your VS Code Terminal output below!")
        
        if not known_face_names:
            print("[ERROR] The face database is completely empty.")
        else:
            print(f"Current Profiles in Database: {known_face_names}")
            name_to_delete = input("Enter the exact name profile you wish to delete: ").strip()
            matched_indices = [i for i, registered_name in enumerate(known_face_names) if registered_name.lower() == name_to_delete.lower()]
            
            if not matched_indices:
                print(f"[ERROR] '{name_to_delete}' does not match any profile.")
            else:
                target_index = matched_indices[0]
                target_name_exact = known_face_names[target_index]
                
                confirm = input(f"[WARNING] Are you sure you want to permanently delete '{target_name_exact}'? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    del known_face_encodings[target_index]
                    del known_face_names[target_index]
                    
                    with open(db_cache_file, "wb") as f:
                        pickle.dump((known_face_encodings, known_face_names), f)
                    print(f"[SUCCESS] System memory updated. Profile '{target_name_exact}' removed.")
                    
                    base_img_format = target_name_exact.lower().replace(' ', '_') + "1"
                    file_purged = False
                    for ext in ['.jpg', '.jpeg', '.png']:
                        potential_file_path = os.path.join(folder_path, base_img_format + ext)
                        if os.path.exists(potential_file_path):
                            os.remove(potential_file_path)
                            file_purged = True
                    
                    if file_purged:
                        print(f"[INFO] Cleaned up associated file '{base_img_format}' from training photos folder.")
                else:
                    print("Deletion process canceled safely.")

    elif key == ord(' '):
        trigger_scan = True
    elif key == ord('q'):
        print("Stopping Live Tracking.")
        break

cap.release()
cv2.destroyAllWindows()