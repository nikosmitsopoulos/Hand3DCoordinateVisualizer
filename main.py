import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.stdout.reconfigure(line_buffering=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    plt.ion()  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("We didn't get a picture from the camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  
                print(f"Detected {label} hand")

                if label == "Left":
                    continue

                lm = hand_landmarks.landmark
                origin = np.array([lm[1].x, lm[1].y, lm[1].z])          # αρχή: βάση αντίχειρα
                index_tip = np.array([lm[5].x, lm[5].y, lm[5].z])        # βάση δείκτη
                thumb_tip = np.array([lm[2].x, lm[2].y, lm[2].z])        # άρθρωση αντίχειρα

                x_axis = index_tip - origin
                x_axis /= np.linalg.norm(x_axis)

                thumb_dir = thumb_tip - origin
                thumb_dir /= np.linalg.norm(thumb_dir)

                z_axis = np.cross(x_axis, np.cross(thumb_dir, x_axis))
                z_axis /= np.linalg.norm(z_axis)

                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                y_axis = -y_axis

                X = origin + x_axis * 0.2
                Y = origin + y_axis * 0.2
                Z = origin + z_axis * 0.2

                ax.cla()
                ax.quiver(*origin, *x_axis, length=0.2, color='r', label='X-axis')
                ax.quiver(*origin, *y_axis, length=0.2, color='g', label='Y-axis')
                ax.quiver(*origin, *z_axis, length=0.2, color='b', label='Z-axis')

                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_zlim([-0.5, 0.5])
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title("Live 3D Coordinate Frame")
                plt.pause(0.001)

                h, w, _ = frame.shape

                def to_pixel_coords(landmark):
                    return np.array([int(landmark.x * w), int(landmark.y * h)])

                
                base_dist = np.linalg.norm(index_tip - origin)
                scale = int(base_dist * w * 2.5)
                O = to_pixel_coords(lm[1])
                X = O + (x_axis[:2] * scale).astype(int)
                Y = O + (y_axis[:2] * scale).astype(int)
                Z = O + (z_axis[:2] * scale).astype(int)

                cv2.line(frame, O, tuple(X), (0, 0, 255), 2)  
                cv2.line(frame, O, tuple(Y), (0, 255, 0), 2) 
                cv2.line(frame, O, tuple(Z), (255, 0, 0), 2)  

                cv2.putText(frame, "X", tuple(X), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Y", tuple(Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Z", tuple(Z), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

