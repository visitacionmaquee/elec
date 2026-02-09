import cv2
import numpy as np
import sys

# -----------------------------
# Load Haar Cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -----------------------------
# Blur Detection
# -----------------------------
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)
# -----------------------------
# Main Classification Function
# -----------------------------
def classify_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    h, w = img.shape[:2]
    center_x = w // 2

    looking_at_camera = False
    off_center = False

    # -----------------------------
    # Face + Eye Analysis
    # -----------------------------
    for (x, y, fw, fh) in faces:
        face_roi_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        # If 2 or more eyes detected, assume facing camera
        if len(eyes) >= 2:
            looking_at_camera = True

        # Composition check
        face_center_x = x + fw // 2
        if abs(face_center_x - center_x) > w * 0.15:
            off_center = True

        # Draw bounding boxes (optional visualization)
        cv2.rectangle(img, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                img[y:y+fh, x:x+fw],
                (ex, ey),
                (ex+ew, ey+eh),
                (255, 0, 0),
                1
            )

    # -----------------------------
    # Blur Analysis
    # -----------------------------
    blur_var = blur_score(img)
    is_blurry = blur_var < 100  # threshold may be tuned

    # -----------------------------
    # Scoring System
    # -----------------------------
    score = 0

    if len(faces) == 0:
        score += 3  # no faces → very likely candid
    if not looking_at_camera:
        score += 2
    if is_blurry:
        score += 1
    if off_center:
        score += 1
    if len(faces) > 1:
        score += 1

    label = "CANDID" if score >= 3 else "POSED"

    # -----------------------------
    # Output
    # -----------------------------
    print("Faces detected:", len(faces))
    print("Looking at camera:", looking_at_camera)
    print("Blur score:", round(blur_var, 2))
    print("Off-center composition:", off_center)
    print("Final score:", score)
    print("Prediction:", label)

    cv2.putText(
        img,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255) if label == "CANDID" else (0, 255, 0),
        3
    )

    show_resized("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


classify_image("candid.jpg")
