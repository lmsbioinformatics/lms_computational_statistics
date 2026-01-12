# Example of statistics and probability
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Random seed
np.random.seed(123)
# Simulate some data
data = np.random.normal(loc=5, scale=1, size=1000)

# Plot data as histogram
plt.figure()
plt.hist(data, bins = 30, alpha = 0.5, edgecolor = "black", linewidth =0.8)
plt.title("Histogram of simulated data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("data_hisotgram.png")

# Plot data as box plot
plt.figure()
sns.boxplot(data)
plt.title("Boxplot of simulated data")
plt.ylabel("Value")
plt.savefig("data_boxplot.png")

# Plot data as violin
plt.figure()
sns.violinplot(data)
plt.title("Violin of simulated data")
plt.ylabel("Value")
plt.savefig("data_violin.png")

# Manual calculation of mean value
def compute_mean(x):
    return sum(x)/len(x)

mean_manual= compute_mean(data)
mean_np = np.mean(data)
print("Mean manual: ", mean_manual)
print("Mean numpy: ", mean_np)

# Manual calculation of variance
def compute_variance(x):
    mean=compute_mean(x)
    squared_diffs= [(x_i-mean_val)**2 for x_i in data]
