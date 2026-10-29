import numpy as np
import matplotlib.pyplot as plt

'''

The Ising model is a simplified computational model that simulates how billions of microscopic atoms interact to create a macroscopic magnet. By representing a material as a grid of individual atomic spins that can only point up or down, the simulation pits two competing forces against each other: the natural desire of neighboring spins to smoothly align with each other (lowering energy), versus the disruptive chaos of heat (temperature). Running this simulation across a spectrum of temperatures ultimately reveals the exact critical tipping point—called a phase transition—where a material abruptly loses its permanent magnetism and transitions from an orderly, functional magnet into a chaotic, non-magnetic state.
'''
N = 50          
 # Lattice size 
T = 2.0          
# Temperature 
J = 1.0          
# Coupling constant 
kb = 1.0         
# Boltzmann constant set to 1 for simplicity
steps = 500000  

# have a n x n array of random choice of up or down 
grid = np.random.choice([-1, 1], size=(N, N))

def get_magnetization(g):
    return np.sum(g) / (N * N)

print(f"Initial Magnetization: {get_magnetization(grid):.3f}")

# Metropolis Algorithm
for step in range(steps):
# spin coords
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)
    
    current_spin = grid[x, y]
    
    # we will sum the  4 nearest neighbors with Periodic Boundary Conditions (using % N)
    neighbors = (grid[(x + 1) % N, y] + 
                 grid[(x - 1) % N, y] + 
                 grid[x, (y + 1) % N] + 
                 grid[x, (y - 1) % N])
    
    # Calculate energy change if this spin were flipped
    delta_E = 2 * J * current_spin * neighbors
    
    # Metropolis acceptance criteria
    if delta_E <= 0:
        # Accept the flip because it lowers or maintains system energy
        grid[x, y] *= -1
    else:
        # Calculate the Boltzmann factor probability
        prob = np.exp(-delta_E / (kb * T))
        # Draw a random float between 0 and 1; accept if it falls under the curve
        if np.random.rand() < prob:
            grid[x, y] *= -1

print(f"Final Magnetization after simulation: {get_magnetization(grid):.3f}")

# 4. Visualize the End State
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='binary', origin='lower')
plt.title(f"Ising Model Final State (T = {T})")
plt.colorbar(label="Spin Direction (+1 or -1)")
plt.show()
