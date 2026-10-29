import numpy as np
import matplotlib.pyplot as plt

'''
so basically an ising model is a mathematical model in statistical mechanics used to describe ferromagnetism (property that
certain materials or elements form permanent magnets or are attracted to them) and phase transitions in materials.

model is basicaly a grid or lattice of atomic sites. each site has a spin variable s_i which is +- 1

Atoms interact with their nearest neighbors

Spins can interact with an external magnetic field

The total energy of the system is determined by summing interactions between neighboring pairs


In a 2d ising model each atom has spin of +- 1
It interacts with its neighbor above or bellow (ie each has 4 interactions (except for the edges?))

Metropolis Algorithm

Monte Carlo simulation solves this by using random sampling. Instead of checking every state, it tracks a single grid and uses a random number generator to intelligently navigate through states that are physically realistic at a given temperature, following the Metropolis Algorithm

metropolis algorithm:

in simple terms basically the algo does:
1) picks a random magnet. pick a random spot in the lattice
2) calculate what would happen to the grids energy if it flipped that magnets direction (from UP to DOWN, or vice versa)

report this as delta e


3) check the energy change
if it lowers the energy (or keeps it the same ) The computer always flips it. (Magnets naturally want to settle into comfortable, low-energy positions).
if it raises the energy: It calculates a probability score based on the temperature
p = e ** -(energy increase/temp)


4) Roll a random number

generates a random number between 0 and 1 
if the number is lower than the score, flips the magnet anyway
if the die roll is higher than the score, leaves alone


= -J Σ s_i s_j (summed over neighboring pairs)
When you flip one spin s, only its 4 bonds change, and those bonds contribute -J·s·n to the total energy, where n is the neighbor sum. After the flip, s becomes -s, so they contribute +J·s·n. The change is
ΔE = (+J·s·n) - (-J·s·n) = 2·J·s·n


what this does
make a n x n grid random assign each site to + 1 or -1

each step rn doing 500,000 times so ~ 200 attempt per site on average

- pick a random site
- sum its 4 neighbors comes out to -4,-2,0,2,4
- calcualte energy change if the spin is flipped 
- if delta is <= 0 flip doesnt raise energy so alawys flip
- if greater than 0 flip anyway with  proabaiblity. if its bellow  that

The final grid, shown as black/white for -1/+1. You'll see clumps (domains) of aligned spins at low T, and noise at high T.

j means coupling strength between neighbors. J > 0 means neighbors prefer to align (ferromagnet), and a bigger J means a stronger preference.

kb is  Boltzmann constant, set to 1 so T is in the same units as J. Only the ratio kbT / J matters.

t is  temperature, which controls how often "bad" flips (ΔE > 0) get accepted. Low T means almost never, so spins lock into aligned domains. High T means often, so thermal noise wins and the grid looks random. The 2D model has a phase transition at T_c ~ 2.27 (in units of J/kb). Below it you get ordered domains, above it disorder.

At T = 2.0 you're just below T_c. Starting from a random grid, it can sometimes freeze into stripe domains instead of one big blob, so the final magnetization can land anywhere from near 0 to near ±1 depending on the run.
'''


def get_magnetization(g):
    return np.sum(g) / g.size


def init_grid(N):
    return np.random.choice([-1, 1], size=(N, N))


def metropolis_step(grid, N, J, kb, T):
    # these are spin coords
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)

    current_spin = grid[x, y]

    #
    neighbors = (grid[(x + 1) % N, y] +
                 grid[(x - 1) % N, y] +
                 grid[x, (y + 1) % N] +
                 grid[x, (y - 1) % N])

    # basicaly summing the spins of neighbors, so the values will be -4,-2,0,2,4

    # energy if spin is flipped
    delta_E = 2 * J * current_spin * neighbors

    if delta_E <= 0:

        grid[x, y] *= -1
        # keep the flipp
    else:
        # calcualte probability and use  Boltzmann
        prob = np.exp(-delta_E / (kb * T))
        # Draw a random float between 0 and 1; accept if it falls under the curve
        if np.random.rand() < prob:
            grid[x, y] *= -1


def run_metropolis(grid, N, J, kb, T, steps):
    # Metropolis Algorithm
    for step in range(steps):
        metropolis_step(grid, N, J, kb, T)
    return grid


def plot_grid(grid, T):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='binary', origin='lower')
    plt.title(f"Ising Model Final State (T = {T})")
    plt.colorbar(label="Spin Direction (+1 or -1)")
    plt.show()


def main():
    N = 50
    T = 2.0
    J = 1.0
    # Coupling constant
    kb = 1.0
    # boltzmann constant set to 1 for now
    steps = 500000

    grid = init_grid(N)

    print(f"Initial Magnetization: {get_magnetization(grid):.3f}")

    run_metropolis(grid, N, J, kb, T, steps)

    print(f"Final Magnetization: {get_magnetization(grid):.3f}")

    plot_grid(grid, T)


if __name__ == '__main__':
    main()
