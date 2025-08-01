from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

def get_data(path):
    df = pd.read_csv(path)
    labels = np.array(df['label'].values)
    probabilities = np.array(df[f'logits_{1}'].values).T
    
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

imdb_base = get_data('../results/IMDB-TextTransformer/test_output_50.csv')
imdb_qama = get_data('../results/IMDB-TextTransformer-QAMA-c_sa/test_output_50.csv')
sst2_base = get_data('../results/SST2-TextTransformer/test_output_50.csv')
sst2_qama = get_data('../results/SST2-TextTransformer-QAMA-c_sa/test_output_50.csv')

plt.figure(figsize=(10, 8))
plt.plot(imdb_base[0], imdb_base[1], color='blue', lw=2, label='IMDB Base ROC curve (AUC = {:.2f})'.format(imdb_base[2]))
plt.plot(imdb_qama[0], imdb_qama[1], color='green', lw=2, label='IMDB QAMA ROC curve (AUC = {:.2f})'.format(imdb_qama[2]))
plt.plot(sst2_base[0], sst2_base[1], color='red', lw=2, label='SST2 Base ROC curve (AUC = {:.2f})'.format(sst2_base[2]))
plt.plot(sst2_qama[0], sst2_qama[1], color='orange', lw=2, label='SST2 QAMA ROC curve (AUC = {:.2f})'.format(sst2_qama[2]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curves')            
plt.legend(loc='lower right', fontsize=16)
plt.grid()
plt.savefig('roc_curves.png', dpi=600, bbox_inches='tight')
plt.savefig('roc_curves.pdf', dpi=600, bbox_inches='tight')
