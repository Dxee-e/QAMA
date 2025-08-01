from tabulate import tabulate
import numpy as np
from matplotlib import pyplot as plt


gurobi_energy = np.load('gurobi_energy.npy')
kaiwu_sa_energy = np.load('kaiwu_sa_energy.npy')
montecarlo_greedy_energy = np.load('montecarlo_greedy_energy.npy')
c_sa_energy = np.load('c_sa_energy.npy')
hybird_energy = np.load('hybird_energy.npy')
energy_matrix = np.stack([gurobi_energy, kaiwu_sa_energy, montecarlo_greedy_energy, c_sa_energy, hybird_energy], axis=1)
methods = ['gurobi', 'kaiwu_sa', 'montecarlo_greedy', 'csa', 'hybird']
min_energy_indices = np.argmin(energy_matrix, axis=1)
min_energy_methods = [methods[idx] for idx in min_energy_indices]

batchsize = gurobi_energy.shape[0]

# table_data = [
#     [i, gurobi_energy[i], kaiwu_sa_energy[i], montecarlo_greedy_energy[i], c_sa_energy[i], hybird_energy[i], min_energy_methods[i]]
#     for i in range(batchsize)
# ]
# print(tabulate(table_data, headers=['Batch', 'Gurobi Energy', 'Kaiwu SA Energy', 'Montecarlo Greedy Energy', 'C++ SA Energy', 'Hybird Energy', 'Best Method'], tablefmt='grid'))


# box plot
diff_gurobi_2_kaiwu_sa = kaiwu_sa_energy - gurobi_energy
diff_gurobi_2_montecarlo_greedy = montecarlo_greedy_energy - gurobi_energy
diff_gurobi_2_c_sa = c_sa_energy - gurobi_energy
diff_gurobi_2_hybird = hybird_energy - gurobi_energy
plt.figure(figsize=(10, 6))
plt.boxplot([diff_gurobi_2_kaiwu_sa, diff_gurobi_2_montecarlo_greedy, diff_gurobi_2_c_sa, diff_gurobi_2_hybird], labels=['Kaiwu SA', 'Montecarlo Greedy', 'CSA', 'Hybird'])
plt.ylim(-200, 600)
plt.title('Energy Difference from Gurobi')
plt.ylabel('Energy Difference')
plt.savefig('energy_difference_boxplot.png')

print("average diff to gurobi:")
print(f"kaiwu_sa: {np.mean(diff_gurobi_2_kaiwu_sa)}")
print(f"montecarlo_greedy: {np.mean(diff_gurobi_2_montecarlo_greedy)}")
print(f"c_sa: {np.mean(diff_gurobi_2_c_sa)}")
print(f"hybird: {np.mean(diff_gurobi_2_hybird)}")

print("diff to kdaiwu_sa diff")
print(f"motecarlo_greedy: {np.mean(diff_gurobi_2_montecarlo_greedy - diff_gurobi_2_kaiwu_sa)}")
print(f"c_sa: {np.mean(diff_gurobi_2_c_sa - diff_gurobi_2_kaiwu_sa)}")
print(f"hybird: {np.mean(diff_gurobi_2_hybird - diff_gurobi_2_kaiwu_sa)}")


best_method_counts = {method: min_energy_methods.count(method) for method in methods}
print("Best method counts:")
for method, count in best_method_counts.items():
    print(f"{method}: {count}")