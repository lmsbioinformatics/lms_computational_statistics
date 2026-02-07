# Introduction to Computational Statistics
# Jesús Urtasun Elizari: MRC LMS 2026
# Chapter 4: Hypothesis testing (II)



# Exercise 4:

# 1. Read control and mutant expression data from the data/ directory

# 2. Plot both distributions as a histogram

# 3. Compute manually a Fisher ANOVA statistic and compare with scipy implementation: Use the stats.f_oneway() library

# 4. Plot the F distribution with the observed F statistic, and a shaded area representing the one-sided p-value

# 5. Plot the F distribution with the observed F statistic, and a shaded area representing the two-sided p-value



# Import libraries ....................................................................................................

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f, f_oneway



# Load data ...........................................................................................................
print("\nLoading data")

# Read csv data with pandas
df_control = pd.read_csv("data/exp_control.csv")
df_mutant1 = pd.read_csv("data/exp_mutant1.csv")
df_mutant2 = pd.read_csv("data/exp_mutant2.csv")

# Extract values as numpy ndarray
control_expr = df_control["avg_expression"].values
mutant1_expr = df_mutant1["avg_expression"].values
mutant2_expr = df_mutant2["avg_expression"].values



# Visual inspection: histogram ........................................................................................

plt.figure(figsize = (8, 5))
plt.hist(control_expr, bins = 30, alpha = 0.4, label = "Control", edgecolor = "black", linewidth = 0.8)
plt.hist(mutant1_expr, bins = 30, alpha = 0.4, label = "Mutant 1", edgecolor = "black", linewidth = 0.8)
plt.hist(mutant2_expr, bins = 30, alpha = 0.4, label = "Mutant 2", edgecolor = "black", linewidth = 0.8)
plt.axvline(np.mean(control_expr), linestyle = "--", linewidth = 2)
plt.axvline(np.mean(mutant1_expr), linestyle = "--", linewidth = 2)
plt.axvline(np.mean(mutant2_expr), linestyle = "--", linewidth = 2)
plt.xlabel("Average expression")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()



# Hypothesis testing (manual implementation) ..........................................................................

print("\nOne-way ANOVA:\nCompare means of three independent groups")

# H0: mean(control) = mean(mutant1) = mean(mutant2)
# H1: at least one group mean differs

groups = [control_expr, mutant1_expr, mutant2_expr]
k = len(groups)
N = sum(len(g) for g in groups)

# Overall mean
overall_mean = np.mean(np.concatenate(groups))

# Between-group sum of squares
SS_between = sum(len(g) * (np.mean(g) - overall_mean)**2 for g in groups)

# Within-group sum of squares
SS_within = sum(sum((g - np.mean(g))**2) for g in groups)

# Degrees of freedom
df_between = k - 1
df_within  = N - k

# Normalized mean squares
MS_between = SS_between / df_between
MS_within  = SS_within / df_within

# F statistic
F_stat = MS_between / MS_within



# One-sided and two-sided p-values (manual) ..........................................................................

# One-sided (standard for ANOVA)
p_one_sided_manual = 1 - f.cdf(F_stat, df_between, df_within)

# Two-sided (shown for symmetry / no physical meaning)
p_two_sided_manual = 2 * min(f.cdf(F_stat, df_between, df_within), 1 - f.cdf(F_stat, df_between, df_within))
print("\nANOVA (manual)")
print(f"\nF statistic = {F_stat:.4f}")
print(f"one-sided p-value = {p_one_sided_manual:.4e}")
print(f"two-sided p-value = {p_two_sided_manual:.4e}")



# Scipy implementation .............................................................................

F_stat_scipy, p_one_sided_scipy = f_oneway(control_expr, mutant1_expr, mutant2_expr)
print("\nANOVA (scipy)")
print(f"F statistic = {F_stat_scipy:.4f}")
print(f"one-sided p-value = {p_one_sided_scipy:.4e}")



# Interpret result (significance level 0.05) ..........................................................................

alpha = 0.05
if p_one_sided_manual < alpha:
    print("\np-value < significance threshold: Reject H0:\nAt least one group mean is significantly different.")
else:
    print("\np-value > significance threshold: Accept H0:\nNo evidence of differences among group means.")



# Plot one-sided p-value ..............................................................................................

# Prepare for plot
x = np.linspace(0.01, F_stat + 3, 1000)
f_dist = f.pdf(x, df_between, df_within)

# Plot F-distribution
plt.figure(figsize=(8, 5))
plt.plot(x, f_dist, label = f"F-distribution (df1={df_between}, df2={df_within})")
plt.axvline(F_stat, color = "red", linestyle="--", label = f"Observed F = {F_stat:.2f}")

# One-tailed rejection region
plt.fill_between(x, f_dist, where = (x >= F_stat), color = "red", alpha = 0.3, label = "One-sided p-value")
plt.xlabel("F value")
plt.ylabel("Density")
plt.title("One-way ANOVA (one-sided)")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("f_test_p_one_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()



# Plot two-sided p-value ..............................................................................................

# Plot F-distribution
plt.figure(figsize=(8, 5))
plt.plot(x, f_dist, label = f"F-distribution (df1={df_between}, df2={df_within})")
plt.axvline(F_stat, color = "red", linestyle = "--", label=f"Observed F = {F_stat:.2f}")

# Two-tailed rejection region
plt.fill_between(x, f_dist, where = (x <= f.ppf(p_two_sided_manual / 2, df_between, df_within)), color = "red", alpha = 0.3)
plt.fill_between(x, f_dist, where = (x >= f.ppf(1 - p_two_sided_manual / 2, df_between, df_within)),color = "red", alpha = 0.3, label = "Two-sided p-value")
plt.xlabel("F value")
plt.ylabel("Density")
plt.title("One-way ANOVA (two-sided, illustrative)")
plt.legend()
plt.grid(alpha = 0.3)
# plt.savefig("f_test_p_two_sided.png", dpi = 300, bbox_inches = "tight")
plt.show()
