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
history_file = "history_log.pkl"

known_face_encodings = []
known_face_names = []

if os.path.exists(db_cache_file):
    print("[INFO] Found cached high-detail database! Loading master profiles...")
    with open(db_cache_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"[SUCCESS] Dynamic profiles loaded. Enrolled users: {known_face_names}")
else:
    print("[INFO] Cache not found. Scanning training directory dynamically...")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    profile_groups = {}
    
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        print(f"[INFO] Discovered {len(files)} image assets. Starting deep mathematical analysis...")
        
        for item in files:
            current_file = os.path.join(folder_path, item)
            
            # Extract and clean identity name directly from the file name string
            base_filename = os.path.splitext(item)[0]
            cleaned_name = base_filename.replace('_', ' ').replace('-', ' ')
            
            # Strip trailing numbers only if they look like file iteration indexes
            while cleaned_name and cleaned_name[-1].isdigit():
                cleaned_name = cleaned_name[:-1]
            identity_name = cleaned_name.strip().title()
            
            try:
                img = face_recognition.load_image_file(current_file)
                face_boxes = face_recognition.face_locations(img, number_of_times_to_upsample=2)
                
                if face_boxes:
                    print(f"[PROCESSING] Deep-analyzing '{identity_name}' from {item} (100 Jitters)...")
                    encoding = face_recognition.face_encodings(img, face_boxes, num_jitters=100, model='large')[0]
                    
                    if identity_name not in profile_groups:
                        profile_groups[identity_name] = []
                    profile_groups[identity_name].append(encoding)
                else:
                    print(f"[WARNING] Could not locate facial matrix boundaries in {item}. Skipping.")
            except Exception as e:
                print(f"[ERROR] Failure processing asset {item}: {e}")
        
        # Build mathematical spatial vector averages for each unique person found
        for name, encodings_list in profile_groups.items():
            master_vector = np.mean(encodings_list, axis=0)
            known_face_encodings.append(master_vector)
            known_face_names.append(name)
            print(f"[MASTER ENROLLED] Identity '{name}' calculated from {len(encodings_list)} source file(s).")
            
        if known_face_encodings:
            with open(db_cache_file, "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
            print("[SUCCESS] Pure dynamic master cache file generated successfully.")
        else:
            print("[WARNING] Extraction pipeline completed with zero profiles stored.")
    else:
        print("[ERROR] Configured training directory path does not exist.")


# =========================================================================
# STEP 1.5: LOAD PERSISTENT DETECTION HISTORY
# =========================================================================
detection_history = []
if os.path.exists(history_file):
    try:
        with open(history_file, "rb") as f:
            detection_history = pickle.load(f)
        print(f"[SUCCESS] Persistent history loaded. Total past records found: {len(detection_history)}")
    except Exception as e:
        print(f"[WARNING] History file reset: {e}")
        detection_history = []


# =========================================================================
# STEP 2: VARIABLE INITIALIZATION & UI CONFIGURATION
# =========================================================================
window_name = 'Interactive Face Recognition'
btn_x1, btn_y1 = 20, 20
btn_x2, btn_y2 = 200, 70
trigger_scan = False  
show_history = False     

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
print("-> Press 'a' key to 'ADD LIVE FACE' from webcam.")
print("-> Press 'f' key to 'ADD FROM FILE/FOLDER' (Single file or bulk upload).")
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

    if trigger_scan:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=5, model='large')
        
        timestamp = time.strftime('%H:%M:%S')
        face_names = []
        
        if not face_encodings:
            detection_history.append(("No Face Detected", timestamp))
            print(f"[{timestamp}] Scan Result: No face detected.")
            
        for face_encoding in face_encodings:
            label = "Unrecognized"
            current_distance = "N/A"
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                current_distance = face_distances[best_match_index]
                
                if current_distance < 0.56:
                    label = known_face_names[best_match_index]
            
            face_names.append(label)
            detection_history.append((label, timestamp))
            print(f"[{timestamp}] Scan Result: {label} (Distance metric: {current_distance})")
            
        with open(history_file, "wb") as f:
            pickle.dump(detection_history, f)
            
        trigger_scan = False

    # --- UI DRAWING ELEMENTS ---
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (230, 100, 10), -1) 
    cv2.putText(frame, "SCAN FACE", (btn_x1 + 25, btn_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, "Keys: 'a'=Add | 'f'=File/Folder | 'd'=Del | 'h'=Toggle History Panel", (20, frame_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 160, 0) if name != "Unrecognized" and name != "No Face Detected" else (0, 0, 170)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if show_history:
        panel_w = 240
        x_start = frame_w - panel_w
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, 0), (frame_w, frame_h), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.line(frame, (x_start, 0), (x_start, frame_h), (70, 70, 70), 1)
        cv2.putText(frame, "SCAN LOG HISTORY", (x_start + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
        cv2.line(frame, (x_start + 15, 45), (frame_w - 15, 45), (100, 100, 100), 1)
        
        y_pos = 80
        recent_logs = detection_history[-8:]
        
        if not recent_logs:
            cv2.putText(frame, "No records found.", (x_start + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        else:
            for log_name, log_time in recent_logs:
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
    
    if key == ord('h'):
        show_history = not show_history
        print(f"\n[TOGGLE] On-screen logs panel viewport visibility set to: {show_history}")

    elif key == ord('a'):
        print("\n[COMMAND] 'Add Live Face' initiated. Analyzing frame...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locs = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        
        if len(temp_locs) == 0:
            print("[ERROR] No face detected in frame.")
        elif len(temp_locs) > 1:
            print("[ERROR] Multiple faces detected. Keep only ONE person in frame.")
        else:
            check_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=5, model='large')[0]
            is_duplicate = False
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.56:
                    print(f"[ABORTED] Face already matches an existing profile: '{known_face_names[best_match_index]}'.")
                    is_duplicate = True
            
            if not is_duplicate:
                print("[INFO] Face verified as unique entry.")
                confirm = input("Would you like to add this live face? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    new_name = input("Enter the name for this new face: ").strip()
                    if new_name:
                        print(f"[INFO] Processing high-fidelity matrix for '{new_name}' (100 Jitters)...")
                        new_encoding = face_recognition.face_encodings(rgb_frame, temp_locs, num_jitters=100, model='large')[0]
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(new_name)
                        
                        # Generalized safe file naming scheme
                        safe_filename = f"{new_name.lower().replace(' ', '_')}.jpg"
                        cv2.imwrite(os.path.join(folder_path, safe_filename), frame)
                        with open(db_cache_file, "wb") as f:
                            pickle.dump((known_face_encodings, known_face_names), f)
                        print(f"[SUCCESS] Added live face for '{new_name}'. Saved as {safe_filename}")
                    else:
                        print("[ERROR] Action canceled. Name cannot be blank.")

    elif key == ord('f'):
        print("\n[COMMAND] 'Add From File/Folder' initiated.")
        print("[INPUT] Look at your VS Code Terminal output below!")
        
        user_path = input("Enter or drag-and-drop the full path to the image file or folder: ").strip().strip("'\"")
        
        if not os.path.exists(user_path):
            print("[ERROR] Provided path does not exist on your computer.")
        else:
            files_to_process = []
            
            if os.path.isfile(user_path):
                files_to_process.append((user_path, True))
            elif os.path.isdir(user_path):
                valid_extensions = ('.jpg', '.jpeg', '.png')
                for item in os.listdir(user_path):
                    if item.lower().endswith(valid_extensions):
                        files_to_process.append((os.path.join(user_path, item), False))
                
                if not files_to_process:
                    print("[ERROR] No valid image assets discovered in that folder.")
            
            success_count = 0
            
            for current_file, prompt_for_name in files_to_process:
                filename_raw = os.path.basename(current_file)
                print(f"\n[PROCESSING] Parsing target file: {filename_raw}")
                
                try:
                    uploaded_img = face_recognition.load_image_file(current_file)
                    temp_locs = face_recognition.face_locations(uploaded_img, number_of_times_to_upsample=2)
                    
                    if len(temp_locs) == 0:
                        print(f"[ERROR] Could not extract any faces from {filename_raw}. Skipping.")
                        continue
                    elif len(temp_locs) > 1:
                        print(f"[ERROR] Multiple faces found in {filename_raw}. Skipping.")
                        continue
                    
                    check_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=5, model='large')[0]
                    is_duplicate = False
                    
                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, check_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.56:
                            print(f"[ABORTED] Face already matches an existing entry: '{known_face_names[best_match_index]}'. Skipping.")
                            is_duplicate = True
                            continue
                    
                    if not is_duplicate:
                        if prompt_for_name:
                            new_name = input("Enter the identity/name for this image file: ").strip()
                        else:
                            base_filename = os.path.splitext(filename_raw)[0]
                            cleaned_name = base_filename.replace('_', ' ').replace('-', ' ')
                            while cleaned_name and cleaned_name[-1].isdigit():
                                cleaned_name = cleaned_name[:-1]
                            new_name = cleaned_name.strip().title()
                            print(f"[INFO] Auto-generated profile label from filename: '{new_name}'")
                        
                        if new_name:
                            print(f"[INFO] Compiling vector matrices for '{new_name}' (100 Jitters)...")
                            new_encoding = face_recognition.face_encodings(uploaded_img, temp_locs, num_jitters=100, model='large')[0]
                            
                            known_face_encodings.append(new_encoding)
                            known_face_names.append(new_name)
                            
                            file_extension = os.path.splitext(current_file)[1]
                            new_filename = f"{new_name.lower().replace(' ', '_')}{file_extension}"
                            destination_target = os.path.join(folder_path, new_filename)
                            
                            if os.path.abspath(current_file) != os.path.abspath(destination_target):
                                shutil.copy(current_file, destination_target)
                            
                            success_count += 1
                        else:
                            print("[ERROR] Input name cannot be empty. Skipping file entry.")
                except Exception as e:
                    print(f"[ERROR] Fatal crash parsing asset data item: {filename_raw}. Reason: {e}")
            
            if files_to_process and success_count > 0:
                with open(db_cache_file, "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                print(f"\n[BATCH COMPLETED] Registered {success_count} new profile record(s) safely.")

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
                    
                    # Look up clean image variants across supported file formats
                    base_img_format = target_name_exact.lower().replace(' ', '_')
                    file_purged = False
                    for ext in ['.jpg', '.jpeg', '.png']:
                        potential_file_path = os.path.join(folder_path, base_img_format + ext)
                        if os.path.exists(potential_file_path):
                            os.remove(potential_file_path)
                            file_purged = True
                    
                    if file_purged:
                        print(f"[INFO] Cleaned up associated file '{base_img_format}' from training folder.")
                else:
                    print("Deletion process canceled safely.")

    elif key == ord(' '):
        trigger_scan = True
    elif key == ord('q'):
        print("Stopping Live Tracking.")
        break

cap.release()
cv2.destroyAllWindows()