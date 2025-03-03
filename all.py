import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables for blink and double-click detection
last_blink_time = 0
blink_count = 0
eye_blink_duration = 0.2  # Time threshold for detecting a blink

# Variables for scrolling
last_eye_position = None

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        # Cursor control: Left Eye Movement (Landmarks 474 to 478)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:  # Landmark for controlling mouse position
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Eye aspect ratio for blinking detection (Left eye)
        left_eye_top = landmarks[159]  # Top point of the left eye
        left_eye_bottom = landmarks[145]  # Bottom point of the left eye
        left_eye_left = landmarks[133]  # Left side of the left eye
        left_eye_right = landmarks[362]  # Right side of the left eye

        # Calculate eye aspect ratio for blink detection
        left_eye_width = left_eye_right.x - left_eye_left.x
        left_eye_height = left_eye_bottom.y - left_eye_top.y
        eye_aspect_ratio = left_eye_height / left_eye_width


        # Detect blink (a blink will cause the aspect ratio to reduce significantly)
        if eye_aspect_ratio < 0.2:
            current_time = time.time()
            if current_time - last_blink_time < 0.3:  # Double-blink detection
                blink_count += 1
            else:
                blink_count = 1
            last_blink_time = current_time

            if blink_count == 2:  # Double click
                pyautogui.doubleClick()
                blink_count = 0
            else:  # Single click
                pyautogui.click()

        # Scroll detection based on vertical eye movement
        if last_eye_position:
            # Compare vertical position of the left eye
            vertical_movement = landmarks[145].y - last_eye_position
            if vertical_movement > 0.02:  # Look down to scroll down
                pyautogui.scroll(-30)
            elif vertical_movement < -0.02:  # Look up to scroll up
                pyautogui.scroll(30)

        # Save the current eye position for next frame's comparison
        last_eye_position = landmarks[145].y

    # Display the video feed
    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing ESC
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
