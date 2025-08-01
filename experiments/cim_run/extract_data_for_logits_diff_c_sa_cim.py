import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
cim = pd.read_csv('cim_output_results.csv')
c_sa = pd.read_csv('c_sa_output_results.csv')

cim_logits = np.where(cim['label']==0, cim['logits_0'], cim['logits_1'])
print(f"Mean CIM logits: {np.mean(cim_logits)}")

c_sa_logits = np.where(c_sa['label']==0, c_sa['logits_0'], c_sa['logits_1'])
print(f"Mean C-SA logits: {np.mean(c_sa_logits)}")

diff = cim_logits - c_sa_logits
print(f"Mean difference: {np.mean(diff)}")
# print(f"Standard deviation of difference: {np.std(diff)}")
# print(f"Max difference: {np.max(diff)}")
# print(f"Min difference: {np.min(diff)}")
