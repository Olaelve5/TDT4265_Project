import os
import cv2
import csv
import numpy as np
from ultralytics import RTDETR

model = RTDETR("runs/detect/RBK_CV/RTDETR_v22/weights/best.pt")
video_path = (
    "/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/aalesund.mp4"
)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out_dir = "aalesund_labeled_v3_2"
os.makedirs(out_dir, exist_ok=True)

output_video_path = os.path.join(out_dir, "aalesund_tracking.mp4")
output_csv_path = os.path.join(out_dir, "tracker_output.csv")

out = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Verdier funnet gjennom tuner.py
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 95, 105])

frame_count = 0

with open(output_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_id", "track_id", "class_id", "x1", "y1", "x2", "y2"])

    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            break

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

        ai_frame = cv2.convertScaleAbs(original_frame, alpha=1.0, beta=50)

        results = model.track(
            ai_frame,
            conf=0.2,
            imgsz=1280,
            agnostic_nms=True,
            persist=True,
            tracker="custom_tracker.yaml",
            verbose=False,
        )

        if results[0].boxes is None:
            out.write(original_frame)
            frame_count += 1
            continue

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            track_id = int(box.id[0]) if box.id is not None else None

            if cls == 0 and conf < 0.60:
                continue
            if cls == 1 and conf < 0.20:
                continue

            # Finn dommere basert på farge på trøye (øvre del av kroppen)
            if cls == 0:
                h = y2 - y1
                y_center = y1 + int(h * 0.50)
                torso_crop = ai_frame[y1:y_center, x1:x2]

                if torso_crop.size == 0:
                    continue

                hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_crop, lower_black, upper_black)

                black_ratio = cv2.countNonZero(mask) / (
                    torso_crop.shape[0] * torso_crop.shape[1]
                )

                if black_ratio > 0.05:
                    continue

            # Logg koordinater
            norm_x1 = x1 / width
            norm_y1 = y1 / height
            norm_x2 = x2 / width
            norm_y2 = y2 / height

            csv_track_id = track_id if track_id is not None else -1
            writer.writerow(
                [frame_count, csv_track_id, cls, norm_x1, norm_y1, norm_x2, norm_y2]
            )

            # Tegn boks og ID på originalrammen
            box_width = x2 - x1
            cx = int((x1 + x2) / 2)

            if cls == 1:
                cy = int((y1 + y2) / 2)
                radius = int(box_width / 2)
                cv2.circle(original_frame, (cx, cy), radius, (0, 0, 255), 2)

                cv2.putText(
                    original_frame,
                    "Ball",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            else:
                axes = (int(box_width / 2), int(box_width * 0.25))
                cv2.ellipse(original_frame, (cx, y2), axes, 0, 0, 360, (255, 0, 0), 2)

                if track_id is not None:
                    cv2.putText(
                        original_frame,
                        f"#{track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

        out.write(original_frame)
        frame_count += 1

cap.release()
out.release()
print(f"Video complete! Both Video and CSV saved in the '{out_dir}/' folder.")
