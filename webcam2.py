import cv2  # For video capture and frame processing
import mediapipe as mp  # For facial landmark detection
import pyautogui  # For mouse control

# Initialize webcam
cam = cv2.VideoCapture(0)  # Opens the default webcam

# Initialize FaceMesh from MediaPipe with refined landmarks
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions for mapping landmarks to screen coordinates
screen_w, screen_h = pyautogui.size()

# Infinite loop to continuously capture video frames
while True:
    _, frame = cam.read()  # Capture a single frame from the webcam
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)

    # Convert the frame to RGB as MediaPipe processes RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect facial landmarks
    output = face_mesh.process(rgb_frame)

    # Get the detected facial landmarks
    landmark_points = output.multi_face_landmarks

    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Check if any landmarks were detected
    if landmark_points:
        # Access landmarks of the first detected face
        landmarks = landmark_points[0].landmark

        # Iterate through specific landmarks corresponding to the right eye (indices 474 to 478)
        for id, landmark in enumerate(landmarks[474:478]):
            # Map landmark coordinates to the video frame dimensions
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw green circles on the right eye landmarks
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Use the second landmark (id == 1) to control the mouse pointer
            if id == 1:
                # Map the landmark coordinates to the screen dimensions
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Move the mouse pointer to the mapped screen coordinates
                pyautogui.moveTo(screen_x, screen_y)

        # Select two landmarks for the left eye (indices 145 and 159)
        left = [landmarks[145], landmarks[159]]

        # Draw circles on the left eye landmarks
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow circles

        # Check if the left eye is blinking (vertical distance between landmarks is small)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()  # Perform a mouse click
            pyautogui.sleep(1)  # Pause to avoid multiple clicks

    # Display the video feed with annotations
    cv2.imshow('Eye Controlled Mouse', frame)

    # Wait for 1ms and check for a key press (keeps the loop running)
    cv2.waitKey(1)
