"""
Example file demonstrating how to make a coloured 2D scatter plot of the maximum
dot products from the main program.
"""
import numpy as np
import matplotlib.pyplot as plt

filename = '../mock_data/dehnendf/main_program_results/dehnendf_psp.npz'
data = np.load(filename)
centres = data['cluster']
values = data['result']

plt.figure(figsize=(20,16))
plt.scatter(*centres.T[:2], c=np.max(np.abs(values), axis=1), marker='.', cmap='plasma')
plt.colorbar(label='Maximum Absolute Dot Product')
plt.xlabel('$x/R_0$')
plt.ylabel('$y/R_0$')
plt.title('Dehnen DF with PowerSphericalPotential, 1.7 Million Stars')
plt.show()