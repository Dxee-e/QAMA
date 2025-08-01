from matplotlib import pyplot as plt
import argparse
import json
import os

args = argparse.ArgumentParser()
args.add_argument('--root_path', type=str)
args.add_argument('--draw_type', nargs='+', type=str, default=['loss', 'acc'])
args = args.parse_args()

VALID_DRAW_TYPES = ['loss', 'acc']
for i in args.draw_type:
    assert i in VALID_DRAW_TYPES, f'Invalid draw type: {i}. Choose from {VALID_DRAW_TYPES}.'

train_record = json.load(open(os.path.join(args.root_path, 'train_record.json'), 'r'))
val_record = json.load(open(os.path.join(args.root_path, 'val_record.json'), 'r'))

metrics_len = len(args.draw_type)
fig, axs = plt.subplots(1, metrics_len, figsize=(6*metrics_len, 5))
for i, metric in enumerate(args.draw_type):
    x = list(range(1, len(train_record[metric])+1))
    axs[i].plot(x, train_record[metric], label='train')
    axs[i].plot(x, val_record[metric], label='val')
    axs[i].set_title(metric)
    axs[i].set_xlabel('epoch')
    axs[i].set_ylabel(metric)
    axs[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(args.root_path, 'train_val_record.png'))
