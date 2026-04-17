import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "/cluster/projects/vc/courses/TDT17/other/Football2025"
CLASS_NAMES = {0: "Player", 1: "Ball", 2: "Event Labels"}

image_extensions = ["*.jpg", "*.jpeg", "*.png"]
all_image_files = []

for ext in image_extensions:
    search_path_img = os.path.join(DATASET_DIR, "**", ext)
    all_image_files.extend(glob.glob(search_path_img, recursive=True))

image_folders = set(os.path.dirname(f) for f in all_image_files)

print(f"-> Success! Found a total of {len(all_image_files)} images.")

print("Searching for text files...")
search_path = os.path.join(DATASET_DIR, "**", "*.txt")
all_txt_files = glob.glob(search_path, recursive=True)

# Filtrer for labels
label_files = []
for f in all_txt_files:
    path_parts = f.lower().split(os.sep)
    filename = os.path.basename(f).lower()

    if "labels" in path_parts and filename not in [
        "classes.txt",
        "train.txt",
        "val.txt",
    ]:
        if os.path.getsize(f) > 0:
            label_files.append(f)


def load_yolo_labels(filepaths):
    data = []
    for file in filepaths:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, cx, cy, w, h = map(float, parts[:5])
                    data.append(
                        {
                            "class_id": int(class_id),
                            "center_x": cx,
                            "center_y": cy,
                            "width": w,
                            "height": h,
                        }
                    )
    return pd.DataFrame(data)


if not label_files:
    print(
        f"Error: Still no label files found. Let's debug what text files actually exist:"
    )
    print(all_txt_files[:5])
    exit()

print(f"Success! Found {len(label_files)} label files.")
df = load_yolo_labels(label_files)
df["class_name"] = df["class_id"].map(CLASS_NAMES).fillna("Unknown")
print(f"Loaded {len(df)} bounding boxes.")

os.makedirs("eda", exist_ok=True)

# Plot 1
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="class_name", order=["Player", "Ball", "Event Labels"])
plt.title("Class Distribution Across All Matches")
plt.xlabel("Class")
plt.ylabel("Number of Bounding Boxes")
plt.savefig("eda/class_distribution.png")
plt.close()
print("Saved eda/class_distribution.png")

# Plot 2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="width", y="height", hue="class_name", alpha=0.5, s=10)
plt.title("Bounding Box Dimensions (Normalized)")
plt.xlabel("Width")
plt.ylabel("Height")
plt.savefig("eda/bbox_dimensions.png")
plt.close()
print("Saved eda/bbox_dimensions.png")
