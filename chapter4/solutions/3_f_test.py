# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 4: Hypothesis testing (II)



# Exercise 3:

# 1. Read control and mutant expression data from the data/ directory

# 2. Plot both distributions as a histogram

# 3. Compute manually a Fisher F statistic and compare with scipy implementation: Use the stats.bartlett() and stats.levene() libraries

# 4. Plot the F distribution with the observed F statistic, and a shaded area representing the one-sided p-value

# 5. Plot the F distribution with the observed F statistic, and a shaded area representing the two-sided p-value



# Import libraries ....................................................................................................

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f, bartlett, levene



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

# Plot histogram
plt.figure(figsize = (8, 5))
plt.hist(control_expr, bins = 30, alpha = 0.5, label = "Control", edgecolor = "black", linewidth = 0.8)
plt.hist(mutant_expr, bins = 30, alpha = 0.5, label = "Mutant", edgecolor = "black", linewidth = 0.8)
plt.axvline(np.mean(control_expr), linestyle = "--", linewidth = 2, label = "Control mean")
plt.axvline(np.mean(mutant_expr), linestyle = "--", linewidth = 2, label = "Mutant mean")
plt.xlabel("Average expression")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha = 0.3)
# plt.savefig("control_vs_mutant_hist.png", dpi=300, bbox_inches="tight")
plt.show()



# Hypothesis testing (manual implementation) ..........................................................................

print("\nF-test:\nCompare variances of two independent groups")

# H0: var(control) = var(mutant)
# H1 one-sided: var(mutant) > var(control)
# H1 two-sided: var(mutant) != var(control)

# Sample variances
var_control = np.var(control_expr, ddof=1)
var_mutant  = np.var(mutant_expr, ddof=1)
n1, n2 = len(control_expr), len(mutant_expr)
print(f"\nVariance control = {var_control:.5f}")
print(f"Variance mutant = {var_mutant:.5f}")

# F-statistic (directional: mutant / control)
F_stat = var_mutant / var_control
df1, df2 = n2 - 1, n1 - 1

# One-sided p-value
p_one_sided_manual = 1 - f.cdf(F_stat, df1, df2)

# Two-sided: symmetric tail probability
p_two_sided_manual = 2 * min(f.cdf(F_stat, df1, df2), 1 - f.cdf(F_stat, df1, df2))

print("\nF-test (manual)")
print(f"one-sided p-value = {p_one_sided_manual:.5f}")
print(f"two-sided p-value = {p_two_sided_manual:.5f}")

# Scipy implementation

# Bartlett's test (likelihood-ratio test)
bart_stat, p_bart = bartlett(control_expr, mutant_expr)
print("\nBartlett test (scipy):")
print(f"B-statistic = {bart_stat:.5f}")
print(f"p-value = {p_bart:.5f}")

# Levene's test (robust, ANOVA-based)
lev_stat, p_lev = levene(control_expr, mutant_expr, center = "mean")
print("\nLevene test (scipy):")
print(f"L-statistic = {lev_stat:.5f}")
print(f"p-value = {p_lev:.5f}")



# Interpret result (significance level 0.05) ..........................................................................
alpha = 0.05
if p_one_sided_manual < alpha:
    print("\np-value < significance threshold: Reject H0:\nExpression in mutant significantly different from expression in control.")
else:
    print("\np-value > significance threshold: Accept H0:\nNo evidence of expression in mutant significantly different than in control.")



# Plot one-sided p-value ............................................................................

# Prepare for plot
x = np.linspace(0.01, F_stat + 2, 1000)
f_dist = f.pdf(x, df1, df2)

# Plot F-distribution
plt.figure(figsize=(8, 5))
plt.plot(x, f_dist, label = f"F-distribution (df1={df1}, df2={df2})")
plt.axvline(F_stat, color = "red", linestyle="--", label = f"Observed F = {F_stat:.2f}")

# One-tailed rejection region
plt.fill_between( x, f_dist, where=(x >= F_stat), color = "red", alpha = 0.3, label = "One-sided p-value")
plt.xlabel("F value")
plt.ylabel("Density")
plt.title("One-sided F-test")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("f_test_p_one_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()



# Plot two-sided p-value ............................................................................

# Plot F-distribution
plt.figure(figsize=(8, 5))
plt.plot(x, f_dist, label=f"F-distribution (df1={df1}, df2={df2})")
plt.axvline(F_stat, color="red", linestyle="--", label=f"Observed F = {F_stat:.2f}")

# Two-tailed rejection region
plt.fill_between(x, f_dist, where=(x <= f.ppf(p_two_sided_manual / 2, df1, df2)), color = "red", alpha = 0.3)
plt.fill_between(x, f_dist, where=(x >= f.ppf(1 - p_two_sided_manual / 2, df1, df2)), color = "red", alpha = 0.3, label = "Two-sided p-value")
plt.xlabel("F value")
plt.ylabel("Density")
plt.title("Two-sided F-test")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("f_test_p_two_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()