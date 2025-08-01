import pandas as pd
import numpy as np

df = pd.read_csv('cim_output_results.csv')

from matplotlib import pyplot as plt

logits = np.where(df['label']==0, df['logits_0'], df['logits_1'])


# plt violin plot
plt.figure(figsize=(10, 6))
plt.violinplot(logits, showmeans=True, showmedians=True)
plt.title('Violin Plot of Logits')
plt.xlabel('Logits')
plt.ylabel('Values')
plt.grid(True)
plt.savefig('violin_plot.png')