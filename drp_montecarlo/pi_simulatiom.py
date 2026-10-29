import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_pi(n):
    num_inside = 0
    in_x_l, in_y_l = [], []
    out_x_l, out_y_l = [], []
    for i in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)

        if x ** 2 + y ** 2 <= 1:
            num_inside += 1
            in_x_l.append(x)
            in_y_l.append(y)
        else:
            out_x_l.append(x)
            out_y_l.append(y)
    pi_estimate = 4 * num_inside / n
    print(f'estimate of pi {pi_estimate} using {n} points. {num_inside} points in the circle out of {n}')
    print(f'percent error {(pi_estimate-np.pi)/np.pi}')

    plt.figure(figsize=(5, 5))
    plt.scatter(in_x_l, in_y_l, s=2, color='green')
    plt.scatter(out_x_l, out_y_l, s=2, color='red')
    plt.gca().add_patch(plt.Circle((0, 0), 1, fill=False, color='black'))
    plt.gca().add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, color='black'))
    plt.title(f'n = {n}, pi ≈ {pi_estimate}')
    plt.gca().set_aspect('equal')
    plt.show()

    return num_inside


def main():
    n_l = [10,100,1000,1000000]

    for i in n_l:
        calculate_pi(i)

if __name__ == '__main__':
    main()