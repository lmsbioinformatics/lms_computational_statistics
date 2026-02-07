# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 1: Descriptive statistics



# Exercise 3: 

# 1. Read expression data from RNA-seq experiment.

# First column represents the gene Ensembl ID.

# Samples labelled '200uM' were grown in 200 micro-molar environment, the control condition.
# Samples labelled '3uM' were grown in 3 micro-molar environment, the deprived condition.
# All samples are labelled 'SL_', indicating they were grown in serum. There are three replicates per each group, labelled '_rep1', '_rep2', '_rep3'.

# Each replicate has one column only, storing raw counts. Counts represent the number reads, i.e. the number of times a transcription of that gene was detected.
# Do not average per replicates, just concatenate / "pool" them with the '.flatten()' function.

# 2. Implement manual computation of mean, median, variance and std.

# 3. Compare with numpy / scipy implementation.

# 4. Compute log transformed counts log(counts + 1) for clear visualization.

# 5. Plot data as a box plot, violin and histogram, and plot summary statistics on top of the histogram.



# Import libraries ............................................................

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load RNA-seq ................................................................

# Read RNA-seq data
rnaseq = pd.read_csv("data/rnaseq_counts.cleaned.csv")
print("\nRNA-seq data\n", rnaseq.head())
print("\nRNA-seq dimensions: ", rnaseq.shape)

# Extract deprived vs control
rnaseq_3uM = rnaseq[[c for c in rnaseq.columns if "3uM" in c]]
rnaseq_200uM = rnaseq[[c for c in rnaseq.columns if "200uM" in c]]

# Remove genes with zero total counts
rnaseq_3uM = rnaseq_3uM[rnaseq_3uM.sum(axis = 1) > 10]
rnaseq_200uM = rnaseq_200uM[rnaseq_200uM.sum(axis = 1) > 10]

# Concatenate replicates, convert 2D matrix to 1D vector
exp_3uM = rnaseq_3uM.values.flatten()
exp_200uM = rnaseq_200uM.values.flatten()



# Compute mean value ..........................................................

# Manual calculation
def compute_mean(data):
    
    return sum(data) / len(data)

# Check implementation
print("\nMean value")
print("3uM (manual):", compute_mean(exp_3uM))
print("3uM (Numpy):", np.mean(exp_3uM))
print("200uM deprived (manual):", compute_mean(exp_200uM))
print("200uM deprived (Numpy):", np.mean(exp_200uM))



# Compute variance ............................................................

# Manual calculation
def compute_var(data):

    mean_val = compute_mean(data)
    squared_diffs = [(x - mean_val) ** 2 for x in data]

    return sum(squared_diffs) / len(data) # population variance
    # return sum(squared_diffs) / (len(data) - 1)  # sample variance

# Check implementation
print("\nVariance")
print("3uM (manual):", compute_var(exp_3uM))
print("3uM (Numpy):", np.var(exp_3uM))
print("200uM deprived (manual):", compute_var(exp_200uM))
print("200uM deprived (Numpy):", np.var(exp_200uM))



# Compute median ..............................................................

# Manual calculation
def compute_median(data):

    # Sort data and find middle point
    sorted_data = sorted(data)
    n = len(sorted_data)
    middle = n // 2 # integer division, divide and round
    
    # Check if even / odd data
    if n % 2 == 0:
        return (sorted_data[middle - 1] + sorted_data[middle]) / 2
    else:
        return sorted_data[middle]

# Check implementation
print("\nMedian")
print("3uM (manual):", compute_median(exp_3uM))
print("3uM (Numpy):", np.median(exp_3uM))
print("200uM deprived (manual):", compute_median(exp_200uM))
print("200uM deprived (Numpy):", np.median(exp_200uM))



# Manual calculation
def compute_std(data):

    return compute_var(data) ** 0.5

# Check implementation
print("\nStandard deviation")
print("3uM (manual):", compute_std(exp_3uM))
print("3uM (Numpy):", np.std(exp_3uM))
print("200uM deprived (manual):", compute_std(exp_200uM))
print("200uM deprived (Numpy):", np.std(exp_200uM))



# Plot RNA-seq data ...........................................................

# Prepare for plot
samples = [exp_3uM, exp_200uM]
labels = ["3uM Deprived", "200uM Control"]

# Log transform for proper visualization
log_3uM = np.log1p(exp_3uM)
log_200uM = np.log1p(exp_200uM)
samples = [log_3uM, log_200uM]
labels = ["3uM Deprived (log)", "200uM Control (log)"]

# Box plot
plt.figure(figsize = (8, 5))
sns.boxplot(data = samples)
plt.xticks(range(2), labels)
plt.ylabel("Expression")
plt.grid(True, alpha = 0.3)
# plt.savefig("rnaseq_box.png", dpi = 300, bbox_inches = "tight")
# plt.show()

# Violin plot
plt.figure(figsize = (8, 5))
sns.violinplot(data = samples)
plt.xticks(range(2), labels)
plt.ylabel("Expression")
plt.grid(True, alpha = 0.3)
# plt.savefig("rnaseq_violin.png", dpi = 300, bbox_inches = "tight")
# plt.show()

# Prepare histogram

# Extract mean per group
m_3uM = compute_mean(samples[0])
m_200uM = compute_mean(samples[1])

# Prepare binsize
all_data = np.concatenate(samples)
bins = np.linspace(all_data.min(), all_data.max(), 30)

# Plot histogram
plt.figure(figsize = (8, 5))
plt.hist(samples[0], bins = bins, label = labels[0], alpha = 0.5, edgecolor = "black", linewidth = 0.8)
plt.hist(samples[1], bins = bins, label = labels[1], alpha = 0.5, edgecolor = "black", linewidth = 0.8)
plt.axvline(m_3uM, color = "blue", linestyle = "--", label = f"3uM mean = {m_3uM:.3f}")
plt.axvline(m_200uM, color = "darkorange", linestyle = "--", label = f"200uM mean = {m_200uM:.3f}")
plt.xlabel("Expression (log)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha = 0.3)
# plt.savefig("rnaseq_histogram.png", dpi = 300, bbox_inches = "tight")
plt.show()
