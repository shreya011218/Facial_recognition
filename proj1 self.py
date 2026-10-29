import face_recognition
import cv2
import numpy as np
import os

# 1. Define the main folder path using a clean raw string
folder_path = r"D:\shreya school\COLLAGE\z Internship\A3i\project\training photos"

print("==================================================")
print("🔍 STEP 1: RUNNING DIAGNOSTICS ON YOUR FOLDER...")
print("==================================================")

# Check if the folder even exists
if not os.path.exists(folder_path):
    print(f"❌ CRITICAL ERROR: The folder path does not exist!")
    print(f"   Please double check if 'COLLAGE' is spelled correctly on your D: drive.")
    exit()

# Print out exactly what files Python sees in that folder
visible_files = os.listdir(folder_path)
print(f"✅ Folder found! Python sees these exact files inside:\n👉 {visible_files}\n")

# 2. Define the exact filenames based on what is in your folder
# WARNING: If your files are named .png or .jpeg, change the extensions below!
images_to_load = {
    "priyanka": "pc1.jpg",
    "alia":     "alia1.jpg",
    "shreya":   "me1.jpg"
}

known_face_encodings = []
known_face_names = []

print("==================================================")
print("📸 STEP 2: LOADING AND ENCODING IMAGES...")
print("==================================================")

for name, filename in images_to_load.items():
    full_path = os.path.join(folder_path, filename)
    
    # Check if individual file exists
    if not os.path.exists(full_path):
        print(f"❌ ERROR: Cannot find the file '{filename}' for {name}!")
        print(f"   Look at the list above—is it spelled differently or missing an extension?")
        continue
        
    # Read the image using OpenCV to guarantee standard 8-bit depth configuration
    img_bgr = cv2.imread(full_path)
    if img_bgr is None:
        print(f"❌ ERROR: OpenCV could not decode '{filename}'. The file might be corrupted.")
        continue
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate facial matrix encodings
    try:
        encodings = face_recognition.face_encodings(img_rgb)
        if len(encodings) == 0:
            print(f"⚠️ WARNING: Found '{filename}', but the AI cannot find a face in it. Make sure it's clear!")
            continue
            
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        print(f"✅ Successfully loaded and encoded: {name}")
        
    except Exception as e:
        print(f"❌ FAILED to encode {name}. Internal Error: {e}")

# Stop if no images loaded successfully
if not known_face_encodings:
    print("\n❌ CRITICAL: No faces were loaded successfully. Fix the paths above and retry.")
    exit()

print("==================================================")
print("🎯 STEP 3: RUNNING THE FACE RECOGNITION TEST...")
print("==================================================")

# 3. Setup the test image (using shreya's photo as the test)
test_filename = "me1.jpg"
test_path = os.path.join(folder_path, test_filename)

if not os.path.exists(test_path):
    print(f"❌ ERROR: Test image '{test_filename}' not found.")
    exit()

test_image = cv2.imread(test_path)
rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_test_image)
face_encodings = face_recognition.face_encodings(rgb_test_image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    print(f"🎯 Result: Found face belonging to -> {name}")

    # Draw bounding boxes and text labels
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 4. Display Window
print("\n🖥️ Opening window... Click on the window and press ANY key to close it.")
cv2.imshow('Face Recognition Result', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Process finished successfully!")