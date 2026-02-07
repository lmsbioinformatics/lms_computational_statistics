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

library(ggplot2)
setwd("/Users/jurtasun/Desktop/courses/LMS/2026/lms_computational_statistics/chapter3/solutions")



# Load EM-seq .................................................................

# Read EM-seq data
emseq <- read.csv("data/emseq_counts.cleaned.csv")
cat("\nEM-seq data\n")
print(head(emseq))
cat("\nEM-seq dimensions:", dim(emseq), "\n")

# Compute methylation score
score <- function(numCs, numTs) {
  numCs / (numCs + numTs + 1e-9)
}

# Average scores for 3uM deprived
s_3uM_rep1 <- score(emseq$X3uM_rep1_numCs, emseq$X3uM_rep1_numTs)
s_3uM_rep2 <- score(emseq$X3uM_rep2_numCs, emseq$X3uM_rep2_numTs)
score_3uM <- (s_3uM_rep1 + s_3uM_rep2) / 2

# Average scores for 200uM control
s_200uM_rep1 <- score(emseq$X200uM_rep1_numCs, emseq$X200uM_rep1_numTs)
s_200uM_rep2 <- score(emseq$X200uM_rep2_numCs, emseq$X200uM_rep2_numTs)
score_200uM <- (s_200uM_rep1 + s_200uM_rep2) / 2



# Compute mean value ..........................................................

# Manual calculation
compute_mean <- function(data) {
  
  sum(data) / length(data)

}

# Check implementation
cat("\nMean value\n")
cat("3uM control (manual):", compute_mean(score_3uM), "\n")
cat("3uM control (R):", mean(score_3uM), "\n")
cat("200uM deprived (manual):", compute_mean(score_200uM), "\n")
cat("200uM deprived (R):", mean(score_200uM), "\n")



# Compute variance ............................................................

# Manual calculation
compute_var <- function(data) {
  
  mean_val <- compute_mean(data)
  squared_diffs <- (data - mean_val) ^ 2
  
  sum(squared_diffs) / length(data)

}

# Check implementation
cat("\nVariance\n")
cat("3uM control (manual):", compute_var(score_3uM), "\n")
cat("3uM control (R):", var(score_3uM) * (length(score_3uM) - 1) / length(score_3uM), "\n")
cat("200uM deprived (manual):", compute_var(score_200uM), "\n")
cat("200uM deprived (R):", var(score_200uM) * (length(score_200uM) - 1) / length(score_200uM), "\n")



# Compute median ..............................................................

# Manual calculation
compute_median <- function(data) {
  
  sorted_data <- sort(data)
  n <- length(sorted_data)
  middle <- n %/% 2
  
  if (n %% 2 == 0) {
    (sorted_data[middle] + sorted_data[middle + 1]) / 2
  } else {
    sorted_data[middle + 1]
  }

}

# Check implementation
cat("\nMedian\n")
cat("3uM control (manual):", compute_median(score_3uM), "\n")
cat("3uM control (R):", median(score_3uM), "\n")
cat("200uM deprived (manual):", compute_median(score_200uM), "\n")
cat("200uM deprived (R):", median(score_200uM), "\n")



# Compute std .................................................................

# Manual calculation
compute_std <- function(data) {

  compute_var(data) ^ 0.5
  
}

# Check implementation
cat("\nStandard deviation\n")
cat("3uM control (manual):", compute_std(score_3uM), "\n")
cat("3uM control (R):", sd(score_3uM) * sqrt((length(score_3uM) - 1) / length(score_3uM)), "\n")
cat("200uM deprived (manual):", compute_std(score_200uM), "\n")
cat("200uM deprived (R):", sd(score_200uM) * sqrt((length(score_200uM) - 1) / length(score_200uM)), "\n")



# Plot data and summary statistics ............................................

df <- data.frame(score = c(score_3uM, score_200uM),
                 condition = rep(c("3uM Deprived", "200uM Control"),
                                 times = c(length(score_3uM), length(score_200uM))))

# Box plot
ggplot(df, aes(x = condition, y = score, fill = condition)) +
  geom_boxplot() + scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(y = "Score") + theme_bw()

# Violin plot
ggplot(df, aes(x = condition, y = score, fill = condition)) +
  geom_violin() + scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(y = "Score") + theme_bw()

# Histogram
ggplot(df, aes(x = score, fill = condition)) +
  geom_histogram(bins = 30, alpha = 0.5, position = "identity", color = "black") +
  geom_vline(xintercept = c(mean(score_3uM), mean(score_200uM)),
             linetype = "dashed", color = c("steelblue", "darkorange")) +
  scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(x = "Score", y = "Frequency") + theme_bw()
