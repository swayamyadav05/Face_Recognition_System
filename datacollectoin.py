import cv2

# Initialize video capture object for laptop camera (assuming index 1)
video = cv2.VideoCapture(1)

# Load pre-trained face detection classifier
detect_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Prompt user to input their ID
id = int(input("Enter your ID: "))

# Initialize counter for captured face images
count = 0

# Main loop for capturing face images
while True:
    # Read a frame from the video capture object
    ret, frame = video.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detect_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for x, y, w, h in faces:
        count += 1

        # Save the detected face region as an image
        cv2.imwrite(f"datasets/User.{id}.{count}.jpg", gray[y : y + h, x : x + w])

        # cv2.imwrite(
        #     "datasets/User." + str(id) + "." + str(count) + ".jpg",
        #     gray[y : y + h, x : x + w],
        # )

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the frame with detected faces
    cv2.imshow("Frame", frame)

    # Wait for a key press
    wait_key = cv2.waitKey(1)

    # Break out of the loop if enough images have been captured
    if count > 500:
        break

# Release the video capture object and close the camera connection
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print a message indicating dataset collection is done
print("Dataset Collection Done............")
