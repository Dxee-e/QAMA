from tabulate import tabulate
import numpy as np
from matplotlib import pyplot as plt


data = np.load('test_error.npz')


gurobi_energy = data['gurobi_quadratic_term'] + data['gurobi_liner_term']
kaiwu_sa_energy = data['kaiwu_sa_quadratic_term'] + data['kaiwu_sa_liner_term']
montecarlo_greedy_energy = data['montecarlo_greedy_quadratic_term'] + data['montecarlo_greedy_liner_term']

energy_matrix = np.stack([gurobi_energy, kaiwu_sa_energy, montecarlo_greedy_energy], axis=1)
methods = ['gurobi', 'kaiwu_sa', 'montecarlo_greedy']
min_energy_indices = np.argmin(energy_matrix, axis=1)
min_energy_methods = [methods[idx] for idx in min_energy_indices]

batchsize = gurobi_energy.shape[0]

table_data = [
    [i, gurobi_energy[i], kaiwu_sa_energy[i], montecarlo_greedy_energy[i], min_energy_methods[i]]
    for i in range(batchsize)
]
print(tabulate(table_data, headers=['Batch', 'Gurobi Energy', 'Kaiwu SA Energy', 'Montecarlo Greedy Energy', 'Best Method'], tablefmt='grid'))


# box plot
diff_gurobi_2_kaiwu_sa = kaiwu_sa_energy - gurobi_energy
diff_gurobi_2_montecarlo_greedy = montecarlo_greedy_energy - gurobi_energy
plt.figure(figsize=(10, 6))
plt.boxplot([diff_gurobi_2_kaiwu_sa, diff_gurobi_2_montecarlo_greedy], labels=['Kaiwu SA', 'Montecarlo Greedy'])
plt.title('Energy Difference from Gurobi')
plt.ylabel('Energy Difference')
plt.savefig('energy_difference_boxplot.png')
