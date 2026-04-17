import os
import glob
import shutil
import yaml

# Samler alle bilder og tilhørende labels i en strukturert mappe klar for trening

DATASET_DIR = "/cluster/projects/vc/courses/TDT17/other/Football2025"
LOCAL_DIR = os.path.abspath("rbk_yolo_data_v2")

if os.path.exists(LOCAL_DIR):
    shutil.rmtree(LOCAL_DIR)

for split in ["train", "val"]:
    os.makedirs(os.path.join(LOCAL_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DIR, "labels", split), exist_ok=True)

# Finn alle label-filer og bygg en rask oppslagstabell for å matche dem med bilder
print("Finding all label files...")
all_labels = glob.glob(os.path.join(DATASET_DIR, "**", "*.txt"), recursive=True)

label_dict = {}
for lp in all_labels:
    if "classes.txt" in lp or "train.txt" in lp or "val.txt" in lp:
        continue

    parts = lp.split(os.sep)
    match_name = next((p for p in parts if p.startswith("RBK-")), "Unknown")

    base_name = os.path.splitext(os.path.basename(lp))[0]

    if match_name not in label_dict:
        label_dict[match_name] = {}
    label_dict[match_name][base_name] = lp

# Finn alle bilder og match dem med labels
print("Finding all images...")
all_images = glob.glob(os.path.join(DATASET_DIR, "**", "*.jpg"), recursive=True)
all_images.extend(glob.glob(os.path.join(DATASET_DIR, "**", "*.png"), recursive=True))

print(f"Found {len(all_images)} raw images! Pairing them up...")

train_count, val_count, missing = 0, 0, 0

for img_path in all_images:
    parts = img_path.split(os.sep)
    match_name = next((p for p in parts if p.startswith("RBK-")), "Unknown")

    if match_name == "RBK-AALESUND":
        split = "val"
    else:
        split = "train"

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    label_path = label_dict.get(match_name, {}).get(base_name)

    if label_path and os.path.getsize(label_path) > 0:
        parent_folder = os.path.basename(os.path.dirname(img_path))
        new_base = f"{match_name}_{parent_folder}_{base_name}"

        new_img_name = f"{new_base}{os.path.splitext(img_path)[1]}"
        new_label_name = f"{new_base}.txt"

        local_img_dest = os.path.join(LOCAL_DIR, "images", split, new_img_name)
        local_label_dest = os.path.join(LOCAL_DIR, "labels", split, new_label_name)

        shutil.copy2(img_path, local_img_dest)
        shutil.copy2(label_path, local_label_dest)

        if split == "train":
            train_count += 1
        else:
            val_count += 1
    else:
        missing += 1

print("-" * 50)
print(f"Training Images: {train_count}")
print(f"Validation Images: {val_count}")
print(f"Total Combined: {train_count + val_count}")
if missing > 0:
    print(f"Skipped {missing} images (No matching .txt label found).")
print("-" * 50)

yaml_content = {
    "path": LOCAL_DIR,
    "train": "images/train",
    "val": "images/val",
    "nc": 2,
    "names": {0: "Player", 1: "Ball"},
}

with open("football2025.yaml", "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print("Data preparation complete!")
