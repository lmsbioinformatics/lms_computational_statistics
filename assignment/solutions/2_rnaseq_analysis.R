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

library(ggplot2)
setwd("/Users/jurtasun/Desktop/courses/LMS/2026/lms_computational_statistics/chapter3/solutions")



# Load RNA-seq ................................................................

# Read RNA-seq data
rnaseq <- read.csv("data/rnaseq_counts.cleaned.csv")
cat("\nRNA-seq data\n")
print(head(rnaseq))
cat("\nRNA-seq dimensions:", dim(rnaseq), "\n")

# Extract deprived vs control
rnaseq_3uM <- rnaseq[, grepl("3uM", colnames(rnaseq))]
rnaseq_200uM <- rnaseq[, grepl("200uM", colnames(rnaseq))]

# Remove genes with zero total counts
rnaseq_3uM <- rnaseq_3uM[rowSums(rnaseq_3uM) > 10, ]
rnaseq_200uM <- rnaseq_200uM[rowSums(rnaseq_200uM) > 10, ]

# Concatenate replicates, convert 2D matrix to 1D vector
exp_3uM <- as.vector(as.matrix(rnaseq_3uM))
exp_200uM <- as.vector(as.matrix(rnaseq_200uM))



# Compute mean value ..........................................................

# Manual calculation
compute_mean <- function(data) {
  
  sum(data) / length(data)

}

# Check implementation
cat("\nMean value\n")
cat("3uM (manual):", compute_mean(exp_3uM), "\n")
cat("3uM (R):", mean(exp_3uM), "\n")
cat("200uM deprived (manual):", compute_mean(exp_200uM), "\n")
cat("200uM deprived (R):", mean(exp_200uM), "\n")



# Compute variance ............................................................

# Manual calculation
compute_var <- function(data) {
  
  mean_val <- compute_mean(data)
  squared_diffs <- (data - mean_val) ^ 2
  
  sum(squared_diffs) / length(data)

}

# Check implementation
cat("\nVariance\n")
cat("3uM (manual):", compute_var(exp_3uM), "\n")
cat("3uM (R):", var(exp_3uM) * (length(exp_3uM) - 1) / length(exp_3uM), "\n")
cat("200uM deprived (manual):", compute_var(exp_200uM), "\n")
cat("200uM deprived (R):", var(exp_200uM) * (length(exp_200uM) - 1) / length(exp_200uM), "\n")



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
cat("3uM (manual):", compute_median(exp_3uM), "\n")
cat("3uM (R):", median(exp_3uM), "\n")
cat("200uM deprived (manual):", compute_median(exp_200uM), "\n")
cat("200uM deprived (R):", median(exp_200uM), "\n")



# Compute std .................................................................

# Manual calculation
compute_std <- function(data) {
  
  compute_var(data) ^ 0.5
  
}

# Check implementation
cat("\nStandard deviation\n")
cat("3uM (manual):", compute_std(exp_3uM), "\n")
cat("3uM (R):", sd(exp_3uM) * sqrt((length(exp_3uM) - 1) / length(exp_3uM)), "\n")
cat("200uM deprived (manual):", compute_std(exp_200uM), "\n")
cat("200uM deprived (R):", sd(exp_200uM) * sqrt((length(exp_200uM) - 1) / length(exp_200uM)), "\n")



# Plot RNA-seq data ...........................................................

# Prepare for plot
log_3uM <- log1p(exp_3uM)
log_200uM <- log1p(exp_200uM)

df <- data.frame(expression = c(log_3uM, log_200uM),
                 condition = rep(c("3uM Deprived (log)", "200uM Control (log)"),
                                 times = c(length(log_3uM), length(log_200uM))))

# Box plot
ggplot(df, aes(x = condition, y = expression, fill = condition)) +
  geom_boxplot() + scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(y = "Expression") + theme_bw()

# Violin plot
ggplot(df, aes(x = condition, y = expression, fill = condition)) +
  geom_violin() + scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(y = "Expression") + theme_bw()

# Histogram
ggplot(df, aes(x = expression, fill = condition)) +
  geom_histogram(bins = 30, alpha = 0.5, position = "identity", color = "black") +
  geom_vline(xintercept = c(mean(log_3uM), mean(log_200uM)),
             linetype = "dashed", color = c("steelblue", "darkorange")) +
  scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(x = "Expression (log)", y = "Frequency") + theme_bw()
