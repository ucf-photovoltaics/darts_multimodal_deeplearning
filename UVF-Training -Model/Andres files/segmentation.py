import os
import cv2

# === Config ===
input_dir = "/mnt/vstor/CSE_MSE_RXF131/staging/sdle/pv-multiscale/UCF-Image-Data/M55/UVF/F0201"
output_dir = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/segmented_cells"
os.makedirs(output_dir, exist_ok=True)

ROWS = 12
COLS = 3

# === Process Each Image ===
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Couldn't load: {fname}")
        continue

    height, width = img.shape[:2]
    cell_h = height // ROWS
    cell_w = width // COLS

    for row in range(ROWS):
        for col in range(COLS):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            cell = img[y1:y2, x1:x2]

            out_name = fname.replace(".jpg", f"_row{row:02}_col{col:02}.jpg")
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, cell)

    print(f"[INFO] Segmented {fname} into {ROWS * COLS} cells.")

print(f"\n✅ Done. Segmented images saved to: {output_dir}")
