import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances

# Import custom trainer and the custom evaluation system
from custom_trainer import CustomTrainer  

# --- Setup Paths ---
base_directory = os.getcwd()

train_json = os.path.join(base_directory, "Annotated/split_data/train.json")
test_json = os.path.join(base_directory, "Annotated/split_data/test.json")
train_img_dir = os.path.join(base_directory, "Annotated/split_data/train")
test_img_dir = os.path.join(base_directory, "Annotated/split_data/test")
output_dir = os.path.join(base_directory, "Training_Outputs")
os.makedirs(output_dir, exist_ok=True)

# --- Register Datasets ---
register_coco_instances("uvf_train", {}, train_json, train_img_dir)
register_coco_instances("uvf_test", {}, test_json, test_img_dir)

# --- Config Setup ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("uvf_train",)
cfg.DATASETS.TEST = ("uvf_test",)
cfg.DATALOADER.NUM_WORKERS = 6

cfg.SOLVER.STEPS = (300000, 500000, 700000)
cfg.SOLVER.GAMMA = 0.1
#cfg.SOLVER.OUTPUT_DIR = "./tensorboard_logs"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.05 # * (cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.MAX_ITER = 1000000
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

cfg.TEST.EVAL_PERIOD = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.MASK_ON = True


# Use pretrained weights unless resuming
weights_path = os.path.join(output_dir, "saved_models", "model_best.pth")
cfg.MODEL.WEIGHTS = weights_path if os.path.exists(weights_path) else model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# --- Start Training ---
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()



# older versions needs to be  cleaned up a tiny bit more

# # Dawn Balaschak 7/12/2025
# # this is the UVF model training system for a detectron2 model for the multimodal system

# import os
# import random
# import cv2
# import json


# #detectron 2 shit
# from detectron2.utils.logger import setup_logger
# #setup_logger()
# # sets up debugg logger

# #spcific things needed
# from detectron2 import model_zoo
# from detectron2.engine import DefaultTrainer
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.data.datasets import register_coco_instances
# from detectron2.utils.visualizer import Visualizer
# from detectron2.engine import DefaultPredictor
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# from detectron2.engine import DefaultTrainer

# #coco checking 
# from pycocotools.coco import COCO

# #  # custom trainier system
# from detectron2.evaluation import COCOEvaluator

# import matplotlib.pyplot as plt




# # # # --- Paths ---
# base_directory = os.getcwd()

# output_dir = os.path.join(base_directory, "Training_Outputs")
# os.makedirs(output_dir, exist_ok=True)


# train_json = r"Annotated/split_data/train.json"
# train_json = os.path.join(base_directory, train_json)


# test_json = r"Annotated/split_data/test.json"
# test_json = os.path.join(base_directory, test_json)


# train_img_dir = r"Annotated/split_data/train"
# train_img_dir = os.path.join(base_directory, train_img_dir)


# test_img_dir = r"Annotated/split_data/test"
# test_img_dir = os.path.join(base_directory, test_img_dir)

# base_images_dir = r"Annotated/Images"
# base_images_dir = os.path.join(base_directory, base_images_dir)


# # --- Register datasets ---
# register_coco_instances("uvf_train", {}, train_json, train_img_dir)
# register_coco_instances("uvf_test", {}, test_json, test_img_dir)

# #evaluator = COCOEvaluator("uvf_test",   tasks=["bbox"], distributed=False,    output_dir=cfg.OUTPUT_DIR)

# class CustomTrainer(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
#         os.makedirs(output_folder, exist_ok=True)
#         return COCOEvaluator(dataset_name, tasks=["bbox"], distributed=False, output_dir=output_folder)

#     def after_step(self):
#         super().after_step()
#         if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
#             print(f"\n Running evaluation at iteration {self.iter + 1}")
#             evaluator = self.build_evaluator(self.cfg, "uvf_test")
#             val_loader = build_detection_test_loader(self.cfg, "uvf_test")
#             results = inference_on_dataset(self.model, val_loader, evaluator)

#             # Save JSON
#             out_json = os.path.join(self.cfg.OUTPUT_DIR, "eval", f"eval_iter_{self.iter+1}.json")
#             with open(out_json, "w") as f:
#                 json.dump(results, f, indent=2)

#             # Optional plot
#             if "bbox" in results and "AP-per-category" in results["bbox"]:
#                 self.save_ap_plot(results["bbox"]["AP-per-category"], self.iter + 1)

#     def save_ap_plot(self, ap_data, iteration):
#         categories = [x[0] for x in ap_data]
#         ap_scores = [x[1] for x in ap_data]

#         plt.figure(figsize=(10, 6))
#         bars = plt.barh(categories, ap_scores)
#         plt.xlabel("Average Precision (AP)")
#         plt.title(f"Per-category AP @ iter {iteration}")
#         plt.tight_layout()

#         plot_path = os.path.join(self.cfg.OUTPUT_DIR, "eval", f"ap_plot_iter_{iteration}.png")
#         plt.savefig(plot_path)
#         plt.close()






# # TRAIN_MODE = True

# # # --- Configuration ---
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("uvf_train",)
# cfg.DATASETS.TEST = ("uvf_test",)
# cfg.DATALOADER.NUM_WORKERS = 2

# # Use pretrained base for first-time training; switch to saved model for resume/eval
# if os.path.exists("/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Training_Outputs/model_final.pth"):
#     cfg.MODEL.WEIGHTS = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Training_Outputs/model_final.pth"
#     print("building from previous model")
# else:
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     print("starting fresh")



# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.MAX_ITER = 50000000
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11            # reset this to be the number of options for classes
# cfg.OUTPUT_DIR = output_dir
# cfg.TEST.EVAL_PERIOD = 100
# cfg.SOLVER.CHECKPOINT_PERIOD = 200

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # adjust for inference
# predictor = DefaultPredictor(cfg)

# # --- Make output dir ---
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)





# # --- Train ---

# evaluator = COCOEvaluator("uvf_test",   tasks=["bbox"], distributed=False,    output_dir=cfg.OUTPUT_DIR)
# val_loader = build_detection_test_loader(cfg, "uvf_test")
# trainer = CustomTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()



# inference_on_dataset(trainer.model, val_loader, evaluator)







# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# predictor = DefaultPredictor(cfg)

# dataset_dicts = DatasetCatalog.get("uvf_test")
# metadata = MetadataCatalog.get("uvf_test")

# for d in random.sample(dataset_dicts, 5):
#     img = cv2.imread(d["file_name"])
#     outputs = predictor(img)

#     v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#     out_path = os.path.join(cfg.OUTPUT_DIR, f"pred_{os.path.basename(d['file_name'])}")
#     cv2.imwrite(out_path, out.get_image()[:, :, ::-1])


# print("compiled properly!")


# # ---- THis will check to make sure the data i sproperly loaded"
# checkLoadedData = False
# if (checkLoadedData):
#     # checkign the detectron the training data set
#     data = DatasetCatalog.get("uvf_train")
#     print(f"Loaded {len(data)} samples from train")
#     print(data[0])  # if exists

    
#     # checking the coco loading
#     coco = COCO(train_json)
#     print(f"Images: {len(coco.imgs)}")
#     print(f"Annotations: {len(coco.anns)}")

#     # Checking if that the images exist aswell for training
#     img_dir = os.path.dirname(train_json) + "/train"
#     bad_imgs = [img['file_name'] for img in coco.loadImgs(coco.getImgIds()) if not os.path.exists(os.path.join(train_img_dir, img['file_name']))]
#     print("Missing images for training:", bad_imgs)


#     # # now to check the testing datasets

#     # checkign the detectron the testing data set
#     data = DatasetCatalog.get("uvf_test")
#     print(f"Loaded {len(data)} samples from testing")
#     print(data[0])  # if exists

    
#     # checking the coco loading
#     coco = COCO(test_json)
#     print(f"Images: {len(coco.imgs)}")
#     print(f"Annotations: {len(coco.anns)}")

#     # Checking if that the images exist aswell    
#     bad_imgs = [img['file_name'] for img in coco.loadImgs(coco.getImgIds()) if not os.path.exists(os.path.join(test_img_dir, img['file_name']))]
#     print("Missing images for testing:", bad_imgs)


#     coco = COCO(train_json)
#     ann = coco.loadAnns(coco.getAnnIds())[0]
#     print(type(ann["segmentation"][0][0]))
#     #exit()