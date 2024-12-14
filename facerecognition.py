import face_recognition
import cv2
import numpy as np

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

# Define the known faces and their corresponding image paths
known_faces = {
    "Arya Dubey": r"C:\Users\hp\OneDrive\Pictures\dubey.jpeg",
    "Arya Dubey": r"C:\Users\hp\OneDrive\Pictures\dubey.jpeg"
}

# Load the known faces and encode them with error handling
print("Loading known faces...")
for name, image_path in known_faces.items():
    try:
        # Load and encode the face
        image = face_recognition.load_image_file(image_path)
        # Convert to RGB if needed
        if image.shape[2] == 4:  # If image has alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3:  # If image is BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)
            print(f"Successfully loaded face for {name}")
        else:
            print(f"No face found in image for {name}")
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")

if not known_face_encodings:
    print("No faces were loaded! Please check your image paths and files.")
    exit()

# Initialize video capture
try:
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open video capture device")
except Exception as e:
    print(f"Error opening video capture: {str(e)}")
    exit()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to exit the program.")

# Process every other frame to improve performance
process_this_frame = True

# Start processing the video frames
while True:
    try:
        # Capture frame from video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Convert the frame to grayscale for OpenCV face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image using OpenCV's Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_names = []

        for (x, y, w, h) in faces:
            # Crop the detected face region
            face = frame[y:y+h, x:x+w]

            # Resize the cropped face for faster processing
            small_face = cv2.resize(face, (0, 0), fx=0.2, fy=0.2)
            rgb_small_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)

            # Get face encoding for this region
            face_encoding = face_recognition.face_encodings(rgb_small_face)
            if face_encoding:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0], tolerance=0.6)
                name = "Unknown"
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding[0]))
                    name = known_face_names[best_match_index]

                face_names.append(name)

                # Draw a box around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x+6, y-6), font, 0.6, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        continue


# Clean up
print("Cleaning up...")
video_capture.release()
cv2.destroyAllWindows()