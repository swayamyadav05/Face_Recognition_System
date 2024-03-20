import cv2

# Initialize cap capture object for laptop camera (assuming index 1)
cap = cv2.capCapture(1)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to recognized IDs
names = ["", "Swayam", "Swayam", "Abhipsha", "Tushar"]


while True:
    # Reads the frame from the cap captured object
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame t grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Load over detected faces
    for x, y, w, h in faces:
        # Predict the ID and confidence of the face
        label_id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Draw filled rectangle for name label background
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)

        # Display name or 'Unknown' based on confidence
        if confidence < 50:
            name = names[label_id]
        else:
            name = "Unknown"

        # Draw name label
        cv2.putText(
            frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for 'e' key press to exit loop
    if cv2.waitKey(25) & 0xFF == ord("e"):
        break

# Release the cap capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Face Recognition Done............")
