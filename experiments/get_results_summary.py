import argparse
import json
import os
import numpy as np

ROUND_NUM = 3
args_parser = argparse.ArgumentParser()
args_parser.add_argument("results_dir", type=str, help="Directory containing results files")
args = args_parser.parse_args()

result_dir = args.results_dir
if not os.path.exists(result_dir):
    raise FileNotFoundError(f"Results directory '{result_dir}' does not exist.")


val_record = json.load(open(os.path.join(result_dir, "val_record.json"), "r"))
val_acc_last = round(val_record["acc"][-1], ROUND_NUM)
last_epoch = len(val_record["acc"]) 
print(f'Val Acc Last {last_epoch}: {val_acc_last}')
val_acc_best = round(max(val_record["acc"]), ROUND_NUM)
best_epoch = np.argmax(val_record["acc"]) + 1
print(f'Val Acc Best {best_epoch}: {val_acc_best}')

