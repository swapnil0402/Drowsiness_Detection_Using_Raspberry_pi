import cv2

# Path to the video file
video_path = 'path/to/your/video.mp4'

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # If the frame was not grabbed (end of the video), break the loop
    if not ret:
        print("End of video")
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' on the keyboard to exit the loop and close the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
