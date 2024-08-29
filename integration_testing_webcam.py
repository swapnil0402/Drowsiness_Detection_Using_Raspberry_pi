import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Initialize Dlib's face detector (HOG-based) and the facial landmark predictor
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 25

output_path = 'output_video.mp4'  # Path to save the output video
recording_duration = 100  # Duration for which recording should continue (in seconds)

cap = cv2.VideoCapture(0)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if can't read
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables
COUNTER = 0
alarm_status = False

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    drowsy_alert = ""
    yawn_alert = ""
    
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear = final_ear(shape)
        distance = lip_distance(shape)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True

                drowsy_alert = "DROWSINESS ALERT!"
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            yawn_alert = "Yawn Alert!"
        else:
            yawn_alert = ""

        # Draw the outline of the eyes and mouth
        left_eye_pts = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        right_eye_pts = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        mouth_pts = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # Draw contours around the eyes
        left_eye = shape[left_eye_pts[0]:left_eye_pts[1]]
        right_eye = shape[right_eye_pts[0]:right_eye_pts[1]]
        mouth = shape[48:60]

        cv2.polylines(frame, [np.int32(left_eye)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [np.int32(right_eye)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [np.int32(mouth)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the eye aspect ratio and lip distance
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Lip Distance: {distance:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display alerts on the frame
    cv2.putText(frame, drowsy_alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, yawn_alert, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Write the frame with alerts to the output video file
    video_writer.write(frame)

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit loop when 'q' is pressed or when the set time duration has passed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time > recording_duration:
        break

# Release video capture and writer objects, and close all OpenCV windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()
