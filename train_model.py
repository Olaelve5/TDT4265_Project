from ultralytics import RTDETR
import time

model = RTDETR("rtdetr-l.pt")
start_time = time.time()

results = model.train(
    data="football2025.yaml",
    epochs=50,
    imgsz=1280,
    batch=4,
    device=0,
    image_weights=True,
    mosaic=0.3,
    scale=0.2,
    hsv_v=0.6,
    hsv_s=0.5,
    project="RBK_CV",
    name="RTDETR_v4",
    workers=8,
    patience=20,
)

end_time = time.time()
compute_hours = (end_time - start_time) / 3600

with open("presentation_metrics.txt", "a") as f:
    f.write(f"--- Training Run ---\n")
    f.write(f"Model: RTDETR\n")
    f.write(f"Batch Size: 4\n")
    f.write(f"Total Compute Time: {compute_hours:.2f} hours\n\n")
