import numpy as np
import random
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)


def monte_carlo_integrate(n, a=0, b=np.pi, f_max=1):
    num_under = 0
    in_x_l, in_y_l = [], []

    out_x_l, out_y_l = [], []

    for i in range(n):
        x = random.uniform(a, b)

        y = random.uniform(0, f_max)

        if y <= f(x):
            num_under += 1
            in_x_l.append(x)

            in_y_l.append(y)
        else:
            out_x_l.append(x)

            out_y_l.append(y)
    box_area = (b - a) * f_max
    integral_estimate = box_area * num_under / n

    exact = 2
    print(f'estimate of integral {integral_estimate} using {n} points. {num_under} points under the curve out of {n}')
    print(f'percent error {(integral_estimate-exact)/exact}')

    plt.figure(figsize=(6, 4))
    plt.scatter(in_x_l, in_y_l, s=2, color='green')
    plt.scatter(out_x_l, out_y_l, s=2, color='red')
    xs = np.linspace(a, b, 500)
    plt.plot(xs, f(xs), color='black')
    plt.gca().add_patch(plt.Rectangle((a, 0), b - a, f_max, fill=False, color='black'))
    plt.title(f'n = {n}, integral ≈ {integral_estimate}')
    plt.show()

    return num_under


def main():
    n_l = [10, 100, 1000, 1000000]

    for i in n_l:
        monte_carlo_integrate(i)


if __name__ == '__main__':
    main()
