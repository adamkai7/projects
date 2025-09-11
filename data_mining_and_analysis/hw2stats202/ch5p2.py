from numpy import *
import matplotlib.pyplot as plt

j = 1  
num_samples = 20000  

n_values = arange(1, num_samples + 1)
probabilities = []

for n in n_values:
    sample = random.choice(range(n), size=n, replace=True)
    probability = mean(sample == j)
    probabilities.append(probability)

# Creating a scatter plot
plt.scatter(n_values, probabilities, s=5)
plt.xlabel('n')
plt.ylabel(f'Probability of {j}th observation in bootstrap sample')
plt.title('Probability of jth observation in bootstrap sample for different values of n')
plt.grid(True)
plt.show()
