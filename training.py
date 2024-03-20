import cv2
import numpy as np
import os

# Load the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset folder
path = "datasets"


# Function to get image IDs and corresponding face images
def get_image_ids_and_faces(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for image_path in image_paths:
        face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if face_image is not None:
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(face_image)
            ids.append(id)
            cv2.imshow("Training", face_image)
            cv2.waitKey(1)
    return ids, faces


# Got image IDs and corresponding face images
ids, faces = get_image_ids_and_faces(path)

# Check if there is data for training
if not ids or not faces:
    print("Error: No data found for training.")
    exit()

# Train the face recognizer
face_recognizer.train(faces, np.array(ids))

# Save the trained model
face_recognizer.save("Trainer.yml")

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print completion message
print("Training Completed........")
