# test_model_load.py
import torch

model_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Case Western Segmentation/model_final.pth"

try:
    print(f"[INFO] Attempting to load model: {model_path}")
    model_data = torch.load(model_path, map_location="cpu")
    print("[SUCCESS] Model loaded successfully.")
    print("[INFO] Top-level keys:", list(model_data.keys()) if isinstance(model_data, dict) else type(model_data))
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
