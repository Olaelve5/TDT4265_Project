import cv2
import numpy as np
from ultralytics import RTDETR

print("Loading Model...")
model = RTDETR("runs/detect/RBK_CV/RTDETR/weights/best.pt")

cap = cv2.VideoCapture("aalesund.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
ret, original_frame = cap.read()

original_frame = cv2.resize(original_frame, (1280, 720))


def nothing(x):
    pass


cv2.namedWindow("AI Tuner")
cv2.createTrackbar("S Upper", "AI Tuner", 95, 255, nothing)
cv2.createTrackbar("Alpha (x10)", "AI Tuner", 10, 30, nothing)
cv2.createTrackbar("Beta", "AI Tuner", 50, 100, nothing)
cv2.createTrackbar("V Upper", "AI Tuner", 105, 255, nothing)
cv2.createTrackbar("Black Ratio %", "AI Tuner", 10, 100, nothing)

print("Running initial AI detection (Please wait)...")
initial_ai_frame = cv2.convertScaleAbs(original_frame, alpha=1.2, beta=40)
results = model.track(
    initial_ai_frame, conf=0.40, imgsz=1280, persist=True, verbose=False
)
latest_boxes = results[0].boxes

print("\n--- GUI Loaded! ---")
print("[SPACEBAR] : Re-run AI with new Alpha/Beta exposure")
print("[Q]        : Quit")

while True:
    alpha = cv2.getTrackbarPos("Alpha (x10)", "AI Tuner") / 10.0
    beta = cv2.getTrackbarPos("Beta", "AI Tuner")
    s_upper = cv2.getTrackbarPos("S Upper", "AI Tuner")
    v_upper = cv2.getTrackbarPos("V Upper", "AI Tuner")
    ratio_thresh = cv2.getTrackbarPos("Black Ratio %", "AI Tuner") / 100.0

    ai_frame = cv2.convertScaleAbs(original_frame, alpha=alpha, beta=beta)

    display_frame = original_frame.copy()
    mask_display = np.zeros(ai_frame.shape[:2], dtype=np.uint8)

    if latest_boxes is not None:
        for box in latest_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            if cls == 0:
                h = y2 - y1
                y_center = y1 + int(h * 0.50)

                torso_crop = ai_frame[y1:y_center, x1:x2]

                if torso_crop.size > 0:
                    hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                    lower_black = np.array([0, 0, 0])
                    upper_black = np.array([180, s_upper, v_upper])
                    mask = cv2.inRange(hsv_crop, lower_black, upper_black)

                    mask_display[y1:y_center, x1:x2] = mask

                    black_ratio = cv2.countNonZero(mask) / (
                        torso_crop.shape[0] * torso_crop.shape[1]
                    )

                    if black_ratio > ratio_thresh:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            display_frame,
                            f"REF {black_ratio:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )
                        continue

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("AI Tuner", display_frame)
    cv2.imshow("Mask Debug", mask_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        print("Running AI on new exposure... (Please wait a second)")
        results = model.track(
            ai_frame, conf=0.40, imgsz=1280, persist=True, verbose=False
        )
        latest_boxes = results[0].boxes
        print("Done! You can tune the color sliders again.")

cv2.destroyAllWindows()
