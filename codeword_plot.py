# this file first reconstructs the codestates from the Pauli spectrums of the states that found by ML, then it calcuates components in the computation basis.
# this is used currently to analyze the code found by ML using the Debbie-loss function
# this file is used to draw figure 2 and 3 in the supplementary material of the paper
import numpy as np
import os
import pickle
import cmath
import matplotlib.pyplot as plt
from utils import parse_qecc_str, make_all_to_all_connection, Pauli_tomography_density_matrix

# Initialize the qecc parameters
qecc_str = '((4,2,1))[3]'  # Specify the desired qecc code
step = 18168 # pair-complementary at log gamma = -1.5
#step = 9548 # self-complementary at log gamma = -1.5
logical_index = 0

tmp0 = parse_qecc_str(qecc_str)
num_qubit = tmp0['num_qubit']
num_logical_dim = tmp0['num_logical_dim']
t = tmp0['weight']
num_layer = tmp0['num_layer']

# Generate connection list
connection_list = make_all_to_all_connection(num_qubit)

# Load the best theta parameters
file_name = f'best_theta_parameters_n{num_qubit}_K{num_logical_dim}_t{t}_L{num_layer}_step{step}.pkl' 
folder = 'parameters'
file_path = os.path.join(folder, file_name)

with open(file_path, 'rb') as file:
    best_theta = pickle.load(file)

#num_logical_dim = 2
rho = Pauli_tomography_density_matrix(logical_index, num_qubit, num_logical_dim, best_theta, num_layer, connection_list)

# Calculate the eigenvalues and eigenvectors of rho
eigenvalues, eigenvectors = np.linalg.eigh(rho)

# Find the index of the largest eigenvalue
max_index = np.argmax(np.abs(eigenvalues))

# Extract the eigenvalue and eigenvector corresponding to the largest eigenvalue
largest_eigenvalue = eigenvalues[max_index]
largest_eigenvector = eigenvectors[:, max_index]

# Print the largest eigenvalue
print("Largest Eigenvalue:", largest_eigenvalue)

# Compute the norm (absolute value) of each entry in the eigenvector
norms = np.abs(largest_eigenvector)

# Calculate the angles of each entry in the eigenvector
angles = [cmath.phase(v) / np.pi for v in largest_eigenvector]

# Print the angles
#print("Angles of each entry in the eigenvector:", angles)

# Create a combined scatter plot
plt.figure(figsize=(6, 4))

# Plot norms in blue
plt.scatter(range(len(norms)), norms, color='b', marker='o', label='Norms')

# Plot angles in orange
plt.scatter(range(len(angles)), angles, color='orange', marker='x', label='Angles')

# Add title and labels
plt.title(f"Norms and phases of logical state {logical_index}")
plt.xlabel("Computational basis component")
plt.ylabel("Value")

# Force the y-axis to be between 0 and 1 for norms and -1 to 1 for angles
plt.ylim(-1, 1)
plt.axhline(0, color='black', lw=0.5, ls='--')  # Add a horizontal line at y=0 for reference

# Add a legend
plt.legend(loc='lower right')
print(norms)

#Save the plot
folder = 'figures'
filename = f'codeword_plot_{logical_index}_step{step}.pdf'
filepath = os.path.join(folder, filename)
plt.savefig(filepath, bbox_inches='tight')

# Display the plot
plt.show()