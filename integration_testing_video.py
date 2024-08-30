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
LED_PIN = 17
GPIO.setup(LED_PIN, GPIO.OUT)

def led_pattern(pattern):
    for duration in pattern:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(duration[0])
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(duration[1])

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    if msg == 'drowsy':
        pattern = [(0.5, 0.5)]  # Longer blink for drowsiness
    elif msg == 'yawn':
        pattern = [(0.1, 0.1)] * 5  # Rapid blinks for yawning
    else:
        pattern = [(0.1, 0.1)]  # Default pattern

    while alarm_status and msg == 'drowsy':
        led_pattern(pattern)
        
    if alarm_status2 and msg == 'yawn':
        saying = True
        led_pattern(pattern)
        saying = False

# (Rest of your code remains unchanged)

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
