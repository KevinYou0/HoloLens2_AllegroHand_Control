import cv2

# Open a connection to the default camera (usually the first camera, index 0)
cap = cv2.VideoCapture(0)
desired_width = 960
desired_height = 1280
# x, y, w, h = 590, 430, 100, 100  # Example values
# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read and display frames in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If the frame is read correctly, ret is True
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv2.imshow('Camera Stream', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
