# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 4: Hypothesis testing (II)



# Exercise 5:

# 1. Read control and mutant expression data from the data/ directory

# 2. Plot both distributions as a histogram

# 3. Compute manually a chi2-statistic and compare with scipy implementation: Use the stats.ttest_ind() library

# 4. Plot the chi2 distribution with the observed chi2 statistic, and a shaded area representing the one-sided p-value

# 5. Plot the chi2 distribution with the observed chi2 statistic, and a shaded area representing the two-sided p-value



# Import libraries ....................................................................................................

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2, chisquare



# Load data ...........................................................................................................
print("\nLoading data")

# Read csv data with pandas
df_control = pd.read_csv("data/exp_control.csv")
df_mutant = pd.read_csv("data/exp_mutant1.csv")
# print("\nControl format:\n", type(df_control))
# print("Control expression:\n", df_control.head())
# print("\nMutant format:\n", type(df_mutant))
# print("Mutant expression:\n", df_mutant.head())

# Extract values as numpy ndarray
control_expr = df_control["avg_expression"].values
mutant_expr = df_mutant["avg_expression"].values
# print("\nControl format:\n", type(control_expr))
# print("Control expression:\n", control_expr[:5])
# print("\nMutant format:\n", type(mutant_expr))
# print("Mutant expression:\n", mutant_expr[:5])



# Visual inspection: histogram ........................................................................................

bins = 30

# Plot histogram
plt.figure(figsize = (8, 5))
plt.hist(control_expr, bins = bins, alpha = 0.5, label = "Control", edgecolor = "black", linewidth = 0.8)
plt.hist(mutant_expr, bins = bins, alpha = 0.5, label = "Mutant", edgecolor = "black", linewidth = 0.8)
plt.axvline(np.mean(control_expr), linestyle = "--", linewidth = 2, label = "Control mean")
plt.axvline(np.mean(mutant_expr), linestyle = "--", linewidth = 2, label = "Mutant mean")
plt.xlabel("Average expression")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha = 0.3)
# plt.savefig("control_vs_mutant_hist.png", dpi=300, bbox_inches="tight")
plt.show()



# Hypothesis testing (manual implementation) ..........................................................................

# Hypothesis testing (manual implementation) ..........................................................................

print("\nchi2 test:\nCompare distributions of two independent groups")

# H0: mutant expression follows the same distribution as control
# H1: mutant expression distribution differs from control

# Define common bin edges
bin_edges = np.histogram_bin_edges(np.concatenate([control_expr, mutant_expr]), bins = bins)
# Observed frequencies: mutant
observed, _ = np.histogram(mutant_expr, bins = bin_edges)
# Expected frequencies: control, scaled to mutant sample size
expected, _ = np.histogram(control_expr, bins = bin_edges)
expected = expected * (observed.sum() / expected.sum())

# Remove bins with zero expected count (chi2 requirement)
mask = expected > 0
observed = observed[mask]
expected = expected[mask]
df = len(observed) - 1



# Manual chi2 statistic ........................................................................................

chi_square_manual = np.sum((observed - expected) ** 2 / expected)
p_value_manual = 1 - chi2.cdf(chi_square_manual, df)
print("\nchi2 test (manual)")
print(f"chi2 statistic = {chi_square_manual:.5f}")
print(f"One-sided p-value = {p_value_manual:.5f}")



# Scipy implementation ...............................................................................................

chi_square_scipy, p_value_scipy = chisquare(observed, f_exp=expected)
print("\nchi2 test (scipy)")
print(f"chi2 statistic = {chi_square_scipy:.5f}")
print(f"One-sided p-value = {p_value_scipy:.5f}")



# Interpret result (significance level 0.05) ..........................................................................

alpha = 0.05
if p_value_manual < alpha:
    print("\np-value < significance threshold: Reject H0:\n"
          "Mutant expression distribution differs from control.")
else:
    print("\np-value > significance threshold: Accept H0:\n"
          "No evidence that mutant distribution differs from control.")



# Plot one-sided p-value ..............................................................................................

# Prepare for plot
x = np.linspace(0, chi_square_manual + 10, 1000)
chi_dist = chi2.pdf(x, df)

# Plot chi2-distribution
plt.figure(figsize = (8, 5))
plt.plot(x, chi_dist, label = f"chi2 distribution (df = {df})")
plt.axvline(chi_square_manual, color = "red", linestyle = "--", label = f"Observed χ² = {chi_square_manual:.2f}")

# One-sided rejection region
plt.fill_between(x, chi_dist, where = (x >= chi_square_manual), color = "red", alpha = 0.3, label = "One-sided p-value")
plt.xlabel("chi2 value")
plt.ylabel("Density")
plt.title("chi2 test (one-sided)")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("chi2_test_p_two_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()



# Plot two-sided p-value (pedagogical symmetry with t-test) ..........................................................

# NOTE: chi2 is naturally one-sided; this is for illustration only

# Plot chi2-distribution
plt.figure(figsize = (8, 5))
plt.plot(x, chi_dist, label = f"chi2 distribution (df = {df})")
plt.axvline(chi_square_manual, color = "red", linestyle="--", label=f"Observed χ² = {chi_square_manual:.2f}")

# One-tailed rejection region
lower = chi2.ppf(p_value_manual / 2, df)
upper = chi2.ppf(1 - p_value_manual / 2, df)
plt.fill_between(x, chi_dist, where = (x <= lower), color = "red", alpha = 0.3)
plt.fill_between(x, chi_dist, where = (x >= upper), color = "red", alpha = 0.3, label = "Two-sided p-value")
plt.xlabel("chi2 value")
plt.ylabel("Density")
plt.title("chi2 test (two-sided, illustrative)")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("chi2_test_p_two_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()
