import numpy as np
import matplotlib.pyplot as plt

# Hyperplane 1: 1 + 3*X1 - X2 = 0
# Hyperplane 2: -2 + X1 + 2*X2 = 0

#THIS IS CHAPTER 9 P 1 FOR QUESTION 6
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
hyperplane1 = 1 + 3 * X1 - X2
hyperplane2 = -2 + X1 + 2 * X2
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, hyperplane1, levels=[0], colors='blue', linewidths=2)
plt.contour(X1, X2, hyperplane2, levels=[0], colors='green', linewidths=2)
plt.fill_between(x1, 1 + 3 * x1, 10, color='red', alpha=0.3, label='1 + 3X1 - X2 > 0')
plt.fill_between(x1, -10, 1 + 3 * x1, color='orange', alpha=0.3, label='1 + 3X1 - X2 < 0')
plt.fill_between(x1, (-2 - x1) / 2, 10, color='red', alpha=0.3, label='-2 + X1 + 2X2 > 0')
plt.fill_between(x1, -10, (-2 - x1) / 2, color='orange', alpha=0.3, label='-2 + X1 + 2X2 < 0')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Hyperplanes and Inequality Regions')
plt.grid(True)
plt.show()
