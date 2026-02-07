# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 4: Hypothesis testing (II)



# Exercise 1:

# 1. Read control and mutant expression data from the data/ directory

# 2. Plot both distributions as a histogram

# 3. Compute manually a two-sample t statistic and compare with scipy implementation: Use the stats.ttest_ind() library

# 4. Plot the t distribution with the observed t statistic, and a shaded area representing the one-sided p-value

# 5. Plot the t distribution with the observed t statistic, and a shaded area representing the two-sided p-value



# Import libraries ....................................................................................................

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind



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

print("\nTwo-sample t-test:\nCompare means of two independent groups")

# H0: mean(control) = mean(mutant)
# H1 one-sided: mean(mutant) >/< mean(control)
# H1 two-sided: mean(mutant) != mean(control)

# Sample statistics
mean1, mean2 = np.mean(control_expr), np.mean(mutant_expr)
s1, s2 = np.std(control_expr, ddof = 1), np.std(mutant_expr, ddof = 1)
n1, n2 = len(control_expr), len(mutant_expr)
print(f"\nMean control = {mean1:.5f}")
print(f"Mean mutant  = {mean2:.5f}")

# Pooled standard deviation (equal-variance assumption)
sp = ((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2)
se  = np.sqrt(sp * (1/n1 + 1/n2))
print(f"Pooled standard error = {se:.5f}")

# Manual t-statistic
t_stat = (mean2 - mean1) / se
df = n1 + n2 - 2

# One-sided p-value
p_one_sided_manual = 1 - t.cdf(t_stat, df)
p_two_sided_manual = 2 * (1 - t.cdf(abs(t_stat), df))
print("\nTwo-sample t-test (manual)")
print(f"t statistic = {t_stat:.5f}")
print(f"one-sided p-value = {p_one_sided_manual:.5f}")
print(f"two-sided p-value (manual) = {p_two_sided_manual:.5f}")

# Scipy implementation
t_stat_scipy, p_two_sided_scipy = ttest_ind(mutant_expr, control_expr, equal_var = True) # Student
# t_stat_scipy, p_two_sided_scipy = ttest_ind(mutant_expr, control_expr, equal_var = False) # Welch (recommended)
p_one_sided_scipy = p_two_sided_scipy / 2 if t_stat_scipy > 0 else 1
print("\nTwo-sample t-test (scipy)")
print(f"t statistic = {t_stat_scipy:.5f}")
print(f"one-sided p-value = {p_one_sided_scipy:.5f}")
print(f"two-sided p-value (scipy)= {p_two_sided_scipy:.5f}")



# Interpret result (significance level 0.05) ..........................................................................
alpha = 0.05
if p_one_sided_manual < alpha:
    print("\np-value < significance threshold: Reject H0:\nExpression in mutant significantly different from expression in control.")
else:
    print("\np-value > significance threshold: Accept H0:\nNo evidence of expression in mutant significantly different than in control.")



# Plot one-sided p-value ..............................................................................................

# Prepare for plot
x = np.linspace(-(t_stat + 5), (t_stat + 5), 1000)
t_dist = t.pdf(x, df)

# Plot t-distribution
plt.figure(figsize=(8, 5))
plt.plot(x, t_dist, label = f"t-distribution (df = {df})")
plt.axvline(t_stat, color = "red", linestyle = "--", label = f"Observed t = {t_stat:.2f}")

# One-tailed rejection region
plt.fill_between(x, t_dist, where=(x >= t_stat), color = "red", alpha = 0.3, label = "One-sided p-value")
plt.xlabel("t value")
plt.ylabel("Density")
plt.title("One-sided t-test")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("t_test_p_one_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()



# Plot two-sided p-value ..............................................................................................

# Plot t-distribution
plt.figure(figsize = (8, 5))
plt.plot(x, t_dist, label = f"t-distribution (df = {df})")
plt.axvline(t_stat, color = "red", linestyle="--", label = f"+t = {t_stat:.2f}")
plt.axvline(-t_stat, color = "red", linestyle="--", label = f"-t = {-t_stat:.2f}")

# Two-tailed rejection region
plt.fill_between(x, t_dist, where = (x >= abs(t_stat)), color = "red", alpha = 0.3)
plt.fill_between(x, t_dist, where=(x <= -abs(t_stat)), color = "red", alpha = 0.3, label = "Two-sided p-value")
plt.xlabel("t value")
plt.ylabel("Density")
plt.title("Two-sided t-test")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("t_test_p_two_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()
