# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 1: Descriptive statistics



# Exercise 2: 

# 1. Read expression data from EM-seq experiment. 

# First column represents the genome sequence coordinates, chr_start_end.

# Samples labelled '200uM' were grown in 200 micro-molar environment, the control condition.
# Samples labelled '3uM' were grown in 3 micro-molar environment, the deprived condition.
# There are two replicates per each group, labelled '_rep1' and '_rep2'.

# Each replicate has a column storing number of Cs (methylated prior to experiment, hence they remain as Cs through the EM-seq experiment).
# Each replicate has a column storing number of Ts (un-metylated, hence converted to Cs through the EM-seq experiment).

# 2. Compute methylation score s = nC / (nC + nT) per replicate individually, then average per replicate.

# 3. Implement manual computation of mean, median, variance and std.

# 4. Compare with numpy / scipy implementation.

# 5. Plot data as a box plot, violin and histogram, and plot summary statistics on top of the histogram.



# Import libraries ............................................................

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load EM-seq .................................................................

# Read EM-seq data
emseq = pd.read_csv("data/emseq_counts.cleaned.csv")
print("\nEM-seq data\n", emseq.head())
print("\nEM-seq dimensions: ", emseq.shape)

# Compute methylation score
def score(numCs, numTs):
    return numCs / (numCs + numTs + 1e-9)

# Average scores for 3uM deprived
s_3uM_rep1 = score(emseq["3uM_rep1_numCs"], emseq["3uM_rep1_numTs"])
s_3uM_rep2 = score(emseq["3uM_rep2_numCs"], emseq["3uM_rep2_numTs"])
score_3uM = (s_3uM_rep1 + s_3uM_rep2) / 2

# Average scores for 200uM control
s_200uM_rep1 = score(emseq["200uM_rep1_numCs"], emseq["200uM_rep1_numTs"])
s_200uM_rep2 = score(emseq["200uM_rep2_numCs"], emseq["200uM_rep2_numTs"])
score_200uM = (s_200uM_rep1 + s_200uM_rep2) / 2



# Compute mean value ..........................................................

# Manual calculation
def compute_mean(data):
    
    return sum(data) / len(data)

# Check implementation
print("\nMean value")
print("3uM control (manual):", compute_mean(score_3uM.values))
print("3uM control (Numpy):", np.mean(score_3uM.values))
print("200uM deprived (manual):", compute_mean(score_200uM.values))
print("200uM deprived (Numpy):", np.mean(score_200uM.values))



# Compute variance ............................................................

# Manual calculation
def compute_var(data):

    mean_val = compute_mean(data)
    squared_diffs = [(x - mean_val) ** 2 for x in data]

    return sum(squared_diffs) / len(data) # population variance
    # return sum(squared_diffs) / (len(data) - 1)  # sample variance

# Check implementation
print("\nVariance")
print("3uM control (manual):", compute_var(score_3uM.values))
print("3uM control (Numpy):", np.var(score_3uM.values))
print("200uM deprived (manual):", compute_var(score_200uM.values))
print("200uM deprived (Numpy):", np.var(score_200uM.values))



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
print("3uM control (manual):", compute_median(score_3uM.values))
print("3uM control (Numpy):", np.median(score_3uM.values))
print("200uM deprived (manual):", compute_median(score_200uM.values))
print("200uM deprived (Numpy):", np.median(score_200uM.values))



# Compute std .................................................................

# Manual calculation
def compute_std(data):
    var = compute_var(data)
    return var ** 0.5


# Check implementation
print("\nStandard deviation")
print("3uM control (manual):", compute_std(score_3uM.values))
print("3uM control (Numpy):", np.std(score_3uM.values))
print("200uM deprived (manual):", compute_std(score_200uM.values))
print("200uM deprived (Numpy):", np.std(score_200uM.values))



# Plot data and summary statistics ............................................

# Prepare for plot
samples = [score_3uM.values, score_200uM.values]
labels = ["3uM Deprived", "200uM Control"]

# Box plot
plt.figure(figsize = (8, 5))
sns.boxplot(data = samples)
plt.xticks(range(2), labels)
plt.ylabel("Score")
plt.grid(True, alpha = 0.3)
# plt.savefig("emseq_box.png", dpi = 300, bbox_inches = "tight")
# plt.show()

# Violin plot
plt.figure(figsize = (8, 5))
sns.violinplot(data = samples)
plt.xticks(range(2), labels)
plt.ylabel("Score")
plt.grid(True, alpha = 0.3)
# plt.savefig("emseq_violin.png", dpi = 300, bbox_inches = "tight")
# plt.show()

# Prepare histogram

# Extract mean per group
mean_3uM = compute_mean(samples[0])
mean_200uM = compute_mean(samples[1])

# Prepare binsize
all_data = np.concatenate(samples)
bins = np.linspace(all_data.min(), all_data.max(), 30)

# Plot histogram
plt.figure(figsize = (8, 5))
plt.hist(samples[0], bins = bins, label = labels[0], alpha = 0.5, edgecolor = "black", linewidth = 0.8)
plt.hist(samples[1], bins = bins, label = labels[1], alpha = 0.5, edgecolor = "black", linewidth = 0.8)
plt.axvline(mean_3uM, color = "blue", linestyle = "--", label = f"3uM mean = {mean_3uM:.3f}")
plt.axvline(mean_200uM, color = "darkorange", linestyle = "--", label = f"200uM mean = {mean_200uM:.3f}")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha = 0.3)
# plt.savefig("emseq_histogram.png", dpi = 300, bbox_inches = "tight")
plt.show()
