import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from matplotlib.font_manager import FontProperties
import csv

font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=9)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = r"pose.mp4"
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("宽度:", width)
print("高度:", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("new_pose.mp4", fourcc, fps, (width, height))

csv_file = open("repl.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow(["frame", "index", "x", "y", "z", "visibility"])

total_v = 0
frame_idx = 0
last = None
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        landmarks = results.pose_landmarks.landmark

        for i, lm in enumerate(landmarks):
            csv_writer.writerow([frame_idx, i, lm.x, lm.y, lm.z, lm.visibility])

        if last == None:
            last = landmarks[25]
        else:
            lk = landmarks[25]
            lh = landmarks[23]
            lfi = landmarks[31]

            ds = (((lk.x - last.x) ** 2 + (lk.y - last.y) ** 2) ** 0.5) * 3.7
            v = ds * fps * 2
            total_v += v
            cv2.putText(image,
                        f"Speed: {v:.2f} m/s",
                        (1000, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
            last = landmarks[25]

        out.write(image)

        cv2.imshow("MediaPipe Feed", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print(f"mean_v:{total_v / frame_idx:.2f}m/s")
    cap.release()
    cv2.destroyAllWindows()
