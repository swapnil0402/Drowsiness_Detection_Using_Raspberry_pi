import cv2
import RPi.GPIO as GPIO
import time

# Setup GPIO
LED_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video stream (0 for the default webcam, or use a video file path)
video_capture = cv2.VideoCapture(0)

# Initialize LED state
led_on = False

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert frame to grayscale (Haar Cascade works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if faces were detected
        if len(faces) > 0:
            if not led_on:
                GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on the LED
                led_on = True
                print("LED ON: Face detected")
        else:
            if led_on:
                GPIO.output(LED_PIN, GPIO.LOW)  # Turn off the LED
                led_on = False
                print("LED OFF: No face detected")

        # Display the resulting frame (optional)
        cv2.imshow('Video', frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
