'''
Ok so this is the file of detectron 2 now I have my version that works but after getting new annotatons 
in a new format it didn't and I was running out of time. This is the script update and done using AI so yea this is vibe code.
I don'tlike doing this but I am short on time so here is what diffrent from mine.

What It Will Do:

✅ Validate your annotation files exist
✅ Register datasets with Detectron2 in COCO format
✅ Extract category information automatically
✅ Train Mask R-CNN with proper configuration
✅ Evaluate on test set with COCO metrics
✅ Visualize predictions on sample images
✅ Run inference on your segmented cell images

'''

import os
import sys
import re
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import cv2
from skimage import io
import time
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Detectron2 imports
import detectron2
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
setup_logger()

random.seed(12345)

# Updated paths for your UVF project
PROJECT_ROOT = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test"
SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "Annotated", "split_data")
TRAIN_IMAGES_PATH = os.path.join(SPLIT_DATA_DIR, "train")
TEST_IMAGES_PATH = os.path.join(SPLIT_DATA_DIR, "test")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
FAILED_PATH = os.path.join(OUTPUTS_PATH, "failed")

# COCO annotation files (your actual split files)
TRAIN_JSON = os.path.join(SPLIT_DATA_DIR, "train.json")
TEST_JSON = os.path.join(SPLIT_DATA_DIR, "test.json")

def validate_coco_files():
    """Validate that COCO annotation files exist"""
    if not os.path.exists(TRAIN_JSON):
        raise FileNotFoundError(f"Train annotations not found: {TRAIN_JSON}")
    if not os.path.exists(TEST_JSON):
        raise FileNotFoundError(f"Test annotations not found: {TEST_JSON}")
    if not os.path.exists(TRAIN_IMAGES_PATH):
        raise FileNotFoundError(f"Train images directory not found: {TRAIN_IMAGES_PATH}")
    if not os.path.exists(TEST_IMAGES_PATH):
        raise FileNotFoundError(f"Test images directory not found: {TEST_IMAGES_PATH}")
    
    print(f"✅ Train annotations: {TRAIN_JSON}")
    print(f"✅ Test annotations: {TEST_JSON}")
    print(f"✅ Train images directory: {TRAIN_IMAGES_PATH}")
    print(f"✅ Test images directory: {TEST_IMAGES_PATH}")

def get_category_info(coco_json_path):
    """Extract category information from COCO file"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data.get('categories', [])
    category_names = [cat['name'] for cat in categories]
    num_classes = len(categories)
    
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Category names: {category_names}")
    
    return num_classes, category_names

def register_uvf_datasets():
    """Register UVF datasets with Detectron2 using COCO format"""
    
    # Clear any existing registrations
    if "uvf_train" in DatasetCatalog:
        DatasetCatalog.remove("uvf_train")
    if "uvf_test" in DatasetCatalog:
        DatasetCatalog.remove("uvf_test")
    
    # Register training dataset
    register_coco_instances(
        "uvf_train",
        {},
        TRAIN_JSON,
        TRAIN_IMAGES_PATH
    )
    
    # Register test dataset
    register_coco_instances(
        "uvf_test", 
        {},
        TEST_JSON,
        TEST_IMAGES_PATH
    )
    
    # Get category information
    num_classes, category_names = get_category_info(TRAIN_JSON)
    
    # Set metadata
    MetadataCatalog.get("uvf_train").set(thing_classes=category_names)
    MetadataCatalog.get("uvf_test").set(thing_classes=category_names)
    
    print("✅ Datasets registered with Detectron2")
    return num_classes, category_names

def setup_config(num_classes, max_iter=3000, learning_rate=0.00025, batch_size=2):
    """Setup Detectron2 configuration"""
    
    cfg = get_cfg()
    
    # Output directory
    cfg.OUTPUT_DIR = OUTPUTS_PATH
    
    # Model configuration - using FPN version for better performance
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("uvf_train",)
    cfg.DATASETS.TEST = ("uvf_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))  # Decay at 70% and 90%
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(500, max_iter // 10)
    
    # Model parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold for better recall
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = max(500, max_iter // 10)  # Evaluate every 10% of training
    
    print(f"🔧 Configuration setup:")
    print(f"   - Max iterations: {max_iter}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    
    return cfg

def train_model(cfg):
    """Train the Mask R-CNN model"""
    
    print("🚂 Starting model training...")
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Initialize and run trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"✅ Training completed in {train_time:.2f} seconds ({train_time/60:.1f} minutes)")
    
    # Return path to trained model
    model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print(f"💾 Model saved to: {model_path}")
    
    return model_path

def evaluate_model(cfg, model_path, score_threshold=0.5):
    """Evaluate the trained model"""
    
    print(f"📏 Evaluating model with score threshold: {score_threshold}")
    
    # Update config for evaluation
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Run COCO evaluation
    evaluator = COCOEvaluator("uvf_test", output_dir=os.path.join(cfg.OUTPUT_DIR, "json"))
    val_loader = build_detection_test_loader(cfg, "uvf_test")
    
    print("Running COCO evaluation...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("📊 Evaluation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return predictor, results

def visualize_predictions(predictor, category_names, num_samples=3, score_threshold=0.5):
    """Visualize model predictions on test images"""
    
    print(f"🖼️  Visualizing {num_samples} random test predictions...")
    
    # Get test dataset
    from detectron2.data import get_detection_dataset_dicts
    dataset_dicts = get_detection_dataset_dicts(["uvf_test"])
    
    if len(dataset_dicts) == 0:
        print("❌ No test images found!")
        return
    
    # Sample random images
    sample_dicts = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for i, d in enumerate(sample_dicts):
        print(f"Processing image {i+1}/{len(sample_dicts)}: {os.path.basename(d['file_name'])}")
        
        # Load image
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"❌ Could not load image: {d['file_name']}")
            continue
        
        # Make prediction
        outputs = predictor(im)
        
        # Visualize
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("uvf_test"),
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        # Draw predictions
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.title(f"Predictions on {os.path.basename(d['file_name'])}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print prediction summary
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)
        if num_detections > 0:
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            print(f"   Detections: {num_detections}")
            for j, (score, cls) in enumerate(zip(scores, classes)):
                class_name = category_names[cls] if cls < len(category_names) else f"Class_{cls}"
                print(f"     {j+1}. {class_name}: {score:.3f}")
        else:
            print(f"   No detections above threshold {score_threshold}")
        print()

def run_inference_on_directory(predictor, input_dir, output_dir, score_threshold=0.5):
    """Run inference on all images in a directory"""
    
    print(f"🔍 Running inference on directory: {input_dir}")
    print(f"💾 Saving results to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to process")
    
    processed = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        
        # Load image
        im = cv2.imread(input_path)
        if im is None:
            print(f"❌ Could not load: {image_file}")
            continue
        
        # Make prediction
        outputs = predictor(im)
        
        # Create visualization
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("uvf_test"),
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save result
        output_filename = f"{os.path.splitext(image_file)[0]}_pred.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert RGB to BGR for cv2.imwrite
        result_image = out.get_image()[:, :, ::-1]
        cv2.imwrite(output_path, result_image)
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(image_files)} images...")
    
    print(f"✅ Inference complete! Processed {processed} images")

def main():
    """Main execution function"""
    
    print("🔬 UVF Solar Module Defect Detection Training")
    print("=" * 50)
    
    try:
        # Step 1: Validate files
        print("\n📋 Step 1: Validating annotation files...")
        validate_coco_files()
        
        # Step 2: Register datasets
        print("\n📊 Step 2: Registering datasets...")
        num_classes, category_names = register_uvf_datasets()
        
        # Step 3: Setup configuration
        print("\n🔧 Step 3: Setting up configuration...")
        cfg = setup_config(
            num_classes=num_classes,
            max_iter=3000,  # Adjust as needed
            learning_rate=0.00025,
            batch_size=2
        )
        
        # Step 4: Train model
        print("\n🚂 Step 4: Training model...")
        model_path = train_model(cfg)
        
        # Step 5: Evaluate model
        print("\n📏 Step 5: Evaluating model...")
        predictor, results = evaluate_model(cfg, model_path, score_threshold=0.5)
        
        # Step 6: Visualize some predictions
        print("\n🖼️  Step 6: Visualizing predictions...")
        visualize_predictions(predictor, category_names, num_samples=3)
        
        # Step 7: Run inference on segmented cells (if they exist)
        segmented_cells_dir = os.path.join(OUTPUTS_PATH, "segmented_cells")
        inference_results_dir = os.path.join(OUTPUTS_PATH, "inference_results")
        
        if os.path.exists(segmented_cells_dir):
            print(f"\n🔍 Step 7: Running inference on segmented cells...")
            run_inference_on_directory(predictor, segmented_cells_dir, inference_results_dir)
        else:
            print(f"\n⚠️  Segmented cells directory not found: {segmented_cells_dir}")
            print("   Skipping cell-level inference")
        
        print("\n🎉 Training and evaluation complete!")
        print(f"📁 Check outputs in: {OUTPUTS_PATH}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure your split data structure exists:")
        print(f"   📁 {SPLIT_DATA_DIR}/")
        print(f"   ├── train.json")
        print(f"   ├── test.json") 
        print(f"   ├── train/ (folder with training images)")
        print(f"   └── test/ (folder with test images)")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()