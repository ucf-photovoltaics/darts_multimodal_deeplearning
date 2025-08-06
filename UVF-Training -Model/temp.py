from collections import OrderedDict
import csv
import os

def log_results_by_task(results, base_filename="results_log", iteration=None):

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

        #print(f"Logged {task} results to '{filename}'")

results =OrderedDict([('bbox', {'AP': 21.64535224700917, 'AP50': 35.92795077283111, 'AP75': 21.767564780757574, 'APs': 21.30826812076812, 'APm': 26.04758350890436, 'APl': 22.243626682459187, 'AR@1': 21.58197662552501, 'AR@10': 26.509173622076855, 'AR@100': 26.509173622076855, 'ARs': 24.660479323308273, 'ARm': 35.42261904761905, 'ARl': 23.322185061315494, 'AP-square': 79.77065766444899, 'AP-ring': 26.521395801214815, 'AP-crack': 3.746803069053708, 'AP-bright_crack': 2.5882479552303055, 'AP-hotspot': 0.0, 'AP-finger_corrosion': 9.713971397139714, 'AP-near_busbar': 58.877887788778885, 'AP-busbar_crack': 26.224657804126277, 'AP-shattered': 0.0, 'AP-misc': 9.009900990099009}), ('segm', {'AP': 19.805672011212447, 'AP50': 32.88847458837217, 'AP75': 18.865586889048043, 'APs': 12.868517641514263, 'APm': 19.70387359240903, 'APl': 22.23498462528157, 'AR@1': 20.63327747521296, 'AR@10': 22.607329229909876, 'AR@100': 22.607329229909876, 'ARs': 16.171679197994987, 'ARm': 22.517857142857146, 'ARl': 23.35637309550353, 'AP-square': 79.37025576809931, 'AP-ring': 22.19169257582669, 'AP-crack': 1.3395457192778104, 'AP-bright_crack': 0.0, 'AP-hotspot': 0.0, 'AP-finger_corrosion': 20.2970297029703, 'AP-near_busbar': 54.95049504950495, 'AP-busbar_crack': 12.184929019217712, 'AP-shattered': 0.0, 'AP-misc': 7.7227722772277225})])

log_results_by_task(results, base_filename="results_log", iteration=1)
# for task_type, metrics in results.items():
#         task_type
#         csv_path = os.path.join(self.eval_dir, f"{task_type}_log.csv")
#         write_header = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as csvfile:
            
#             for metric, value in metrics.items():
#                 #print(f"{metric}: {value:.2f}")
                
#                     writer = csv.writer(csvfile)
#                     if write_header:
#                         writer.writerow([1, ])
            
    