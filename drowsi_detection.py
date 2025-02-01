import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import winsound

# Constants for drowsiness detection
frequency = 2500
duration = 1000
earThresh = 0.3  # EAR threshold
earFrames = 48  # Consecutive frames threshold
count = 0

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR)
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmarks indices for Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start video capture
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using Mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates of the left and right eyes
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            # Convert normalized coordinates to pixel values
            left_eye_coords = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in left_eye]
            right_eye_coords = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in right_eye]

            # Calculate EAR for both eyes
            leftEAR = eyeAspectRatio(left_eye_coords)
            rightEAR = eyeAspectRatio(right_eye_coords)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye landmarks
            for point in left_eye_coords:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye_coords:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Check if EAR is below the threshold
            if ear < earThresh:
                count += 1

                # If drowsiness is detected for enough frames
                if count >= earFrames:
                    cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(frequency, duration)
            else:
                count = 0

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
