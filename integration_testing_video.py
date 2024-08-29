import RPi.GPIO as GPIO
import time
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread

# GPIO setup
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 17
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def buzzer_pattern(pattern):
    for duration in pattern:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration[0])
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(duration[1])

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    if msg == 'drowsy':
        pattern = [(0.5, 0.5)]  # Longer beep for drowsiness
    elif msg == 'yawn':
        pattern = [(0.1, 0.1)] * 5  # Short beeps for yawning
    else:
        pattern = [(0.1, 0.1)]  # Default pattern

    while alarm_status and msg == 'drowsy':
        buzzer_pattern(pattern)
        
    if alarm_status2 and msg == 'yawn':
        saying = True
        buzzer_pattern(pattern)
        saying = False

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

# Initialize Dlib's face detector and the facial landmark predictor
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 30

video_path = 'Integration_testing_data/sriya6.mp4'
output_path = 'Integration_testing_output/output_video_sriya66.mp4'  # Path to save the output video

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False

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
                    t = Thread(target=alarm, args=('drowsy',))
                    t.daemon = True
                    t.start()

                drowsy_alert = "DROWSINESS ALERT!"
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            yawn_alert = "Yawn Alert!"
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('yawn',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        # Draw the outline of the eyes and mouth
        left_eye_pts = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        right_eye_pts = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        mouth_pts = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # Draw contours around the eyes and mouth
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
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and clean up GPIO
cap.release()
video_writer.release()
cv2.destroyAllWindows()
GPIO.cleanup()
