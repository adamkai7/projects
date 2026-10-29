import numpy as np
import random
import matplotlib.pyplot as plt

# P[i][j] = probability of going from state i+1 to state j+1
P = np.array([[1/2, 1/4, 1/4],
              [1/4, 1/4, 1/2],
              [0, 1, 0]])


def random_walk(n, start=0):
    state = start
    counts = np.zeros(3)

    running = []
    for i in range(n):
        print(f'tmp')

    return counts


def main():
    n_l = [10, 100, 1000, 1000000]

    for i in n_l:
        random_walk(i)


if __name__ == '__main__':
    main()
