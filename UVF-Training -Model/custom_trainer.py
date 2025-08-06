import os
import json
import csv
import random
import cv2
import io
import contextlib

import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from pycocotools.cocoeval import COCOeval

from detectron2.evaluation.coco_evaluation import create_small_table
import numpy as np
import itertools
from tabulate import tabulate

#from custom_coco_evaluation import COCOEvaluator
# custom coco evaluation for tracking the AR, for future use
class CustomCOCOEvaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        # adjusted by Dawn this makes it so that it sends back the Average Recall aswell
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_ap = 0.0  # Track best AP
        self.eval_dir = os.path.join(cfg.OUTPUT_DIR, "eval")
        self.checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "saved_models")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return CustomCOCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)
        #return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def build_checkpointer(cls, cfg, model, optimizer, scheduler):
        checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "saved_models")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return DetectionCheckpointer(model, checkpoint_dir, optimizer=optimizer, scheduler=scheduler)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        # Remove default checkpointer hook
        hooks_list = [h for h in hooks_list if not isinstance(h, hooks.PeriodicCheckpointer)]

        # Add custom checkpointer saving to "saved_models"
        custom_checkpointer = DetectionCheckpointer(
            self.model,
            save_dir=os.path.join(self.cfg.OUTPUT_DIR, "saved_models"),  # <-- new save path
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        hooks_list.insert(-1, hooks.PeriodicCheckpointer(  # -1 = before EvalHook
            checkpointer=custom_checkpointer,
            period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
            max_to_keep=5
        ))

        return hooks_list

    def log_evaluation_metrics(self, results, iteration):
        # --- Text log ---
        # log_path = os.path.join(self.eval_dir, f"metrics_iter_{iteration}.txt")
        # with open(log_path, "w") as f:
        #     f.write(f"Evaluation Results at Iteration {iteration}\n")
        #     f.write("="*50 + "\n")

        #     general_metrics = {}
        #     for key, value in results.get("bbox", {}).items():
        #         if isinstance(value, float) and not key.startswith("AP-"):
        #             general_metrics[key] = value
        #             f.write(f"{key:40s}: {value:.3f}\n")

        #     # Per-category AP logging
        #     f.write("\nPer-category bbox AP:\n")
        #     f.write("-"*50 + "\n")
        #     category_metrics = {}
        #     per_cat = results["bbox"].get("AP-per-category", [])
        #     if per_cat:
        #         for category, ap in per_cat:
        #             category_metrics[category] = ap
        #             f.write(f"{category:25s}: {ap:.3f}\n")
        #     else:
        #         f.write("No per-category AP results available.\n")

        # # --- CSV log ---
        # csv_path = os.path.join(self.eval_dir, "ap_log.csv")
        # fieldnames = ["iteration"] + list(general_metrics.keys()) + list(category_metrics.keys())

        # row = {"iteration": iteration}
        # row.update(general_metrics)
        # row.update(category_metrics)

        # write_header = not os.path.exists(csv_path)
        # with open(csv_path, "a", newline="") as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #     if write_header:
        #         writer.writeheader()
        #     writer.writerow(row)


        # probably the better one
        log_path = os.path.join(self.eval_dir, f"metrics_iter_{iteration}.txt")
        with open(log_path, "w") as f:
            f.write(f"Evaluation Results at Iteration {iteration}\n")
            f.write("="*50 + "\n")

            # Log AP/AR summary
            for key, value in results.get("bbox", {}).items():
                if isinstance(value, float):
                    f.write(f"{key:40s}: {value:.3f}\n")

            f.write("\nPer-category bbox AP:\n")
            f.write("-"*50 + "\n")
            for cat_ap in results["bbox"].get("AP-per-category", []):
                category, ap = cat_ap
                f.write(f"{category:25s}: {ap:.3f}\n")


    def after_step(self):
        super().after_step()
        if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
            print(f"\n📊 Running evaluation at iteration {self.iter + 1}")
            evaluator = self.build_evaluator(self.cfg, "uvf_test", output_folder=self.eval_dir)
            val_loader = build_detection_test_loader(self.cfg, "uvf_test")
            results = inference_on_dataset(self.model, val_loader, evaluator)

            # Save JSON result for individualized data
            out_json = os.path.join(self.eval_dir, f"eval_iter_{self.iter+1}.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)

            # store data for the long term log
            self.log_to_csv(results, self.iter + 1)

            

            # Save best model if improved
            if "segm" in results:
                ap = results["segm"].get("AP", 0.0)
                print(f"got the results {ap}")
                if ap > self.best_ap:
                    print(f"💾 New best AP: {ap:.3f} (prev {self.best_ap:.3f}) — saving model_best.pth")
                    self.best_ap = ap
                    self.checkpointer.save("model_best")
                #print("saved results are:")
                #print(results)
                #exit()


            # Optional: save visualizations needs to be worked on
            # Save bar plot for later?
            # if "bbox" in results and "AP-per-category" in results["bbox"]:
            #     self.save_ap_plot(results["bbox"]["AP-per-category"], self.iter + 1)

            #self.log_evaluation_metrics(results, self.iter + 1)
            
            #self.save_sample_predictions(self.iter + 1)

    def save_ap_plot(self, ap_data, iteration):
        categories = [x[0] for x in ap_data]
        ap_scores = [x[1] for x in ap_data]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(categories, ap_scores)
        plt.xlabel("Average Precision (AP)")
        plt.title(f"Per-category AP @ iter {iteration}")
        plt.tight_layout()

        plot_path = os.path.join(self.eval_dir, f"ap_plot_iter_{iteration}.png")
        plt.savefig(plot_path)
        plt.close()


    def log_to_csv(self, results, iteration):


        for task, metrics in results.items():
            # Prepare filename: e.g., results_log_bbox.csv
            csv_path = os.path.join(self.eval_dir, f"{task}_AP_traing_log.csv")
            
            # Prepare flat row with optional iteration column
            row = {"iteration": iteration} if iteration is not None else {}
            row.update(metrics)

            # Check if file exists
            file_exists = os.path.isfile(csv_path)

            # Write to CSV
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
       
    def save_sample_predictions(self, iteration, num_samples=3):
        dataset_dicts = DatasetCatalog.get("uvf_test")
        raw_sample = random.choice(dataset_dicts)

        mapper = DatasetMapper(self.cfg, is_train=False)
        processed_sample = mapper(raw_sample)

        inputs = [processed_sample]
        outputs = self.model(inputs)[0]

        img = utils.read_image(raw_sample["file_name"], format="BGR")
        metadata = MetadataCatalog.get("uvf_test")

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        out_path = os.path.join(self.eval_dir, f"sample_pred_iter_{iteration}.png")
        cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])
        # dataset_dicts = DatasetCatalog.get("uvf_test")
        # metadata = MetadataCatalog.get("uvf_test")
        # for i, d in enumerate(random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))):
        #     img = cv2.imread(d["file_name"])
        #     outputs = self.model([d])[0]
        #     v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     out_path = os.path.join(self.eval_dir, f"sample_{iteration}_{i}.jpg")
        #     cv2.imwrite(out_path, out.get_image()[:, :, ::-1])
