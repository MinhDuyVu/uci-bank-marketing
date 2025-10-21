library(tidyverse)
library(caret)
library(smotefamily)
library(pROC)

set.seed(42)

train <- read.csv("data/train_prepared.csv")
val   <- read.csv("data/validation_prepared.csv")
test  <- read.csv("data/test_prepared.csv")

# 1. Feature Selection
cat("\nFeature count before selection:", ncol(train), "\n")

# Remove near-zero variance
nzv <- nearZeroVar(train)
if (length(nzv) > 0) {
  train <- train[, -nzv]
  val <- val[, -nzv]
  test <- test[, -nzv]
}

cat("Feature count after removing near-zero variance:", ncol(train), "\n")

# Remove highly correlated features
numeric_cols <- sapply(train, is.numeric)
corr_mat <- cor(train[, numeric_cols], use = "pairwise.complete.obs")
high_corr <- findCorrelation(corr_mat, cutoff = 0.95)
if (length(high_corr) > 0) {
  numeric_col_names <- names(train)[numeric_cols]
  cols_to_remove <- numeric_col_names[high_corr]
  
  train <- train[, !names(train) %in% cols_to_remove]
  val <- val[, !names(val) %in% cols_to_remove]
  test <- test[, !names(test) %in% cols_to_remove]
}

cat("Feature count after removing highly correlated features:", ncol(train), "\n")

# Verify no mismatch
cat("\nColumns in train but not in val:", 
    paste(setdiff(names(train), names(val)), collapse = ", "), "\n")
cat("Columns in val but not in train:", 
    paste(setdiff(names(val), names(train)), collapse = ", "), "\n")

# 2. Handling Class Imbalance
cat("\nOriginal class distribution (train):\n")
print(table(train$y))
cat("Positive rate:", round(mean(train$y == 1) * 100, 2), "%\n")

# Extract target and features AFTER feature selection
target <- train$y
X <- train %>% select(-y)

# Calculate dup_size
minority_count <- sum(target == 1)
majority_count <- sum(target == 0)
dup_size_needed <- ceiling((majority_count * 0.6 / minority_count) - 1)

cat("\nMinority class:", minority_count, "\n")
cat("Majority class:", majority_count, "\n")
cat("dup_size needed for balance:", dup_size_needed, "\n")

# Apply SMOTE
smote_out <- SMOTE(X = X, target = target, K = 5, dup_size = dup_size_needed)

train_smote <- smote_out$data %>%
  rename(y = class)

cat("\nClass distribution after SMOTE:\n")
print(table(train_smote$y))
cat("Positive rate:", round(mean(train_smote$y == 1) * 100, 2), "%\n")

# Final verification
cat("\nFinal column check:\n")
cat("Columns in train_smote but not in val:", 
    paste(setdiff(names(train_smote), names(val)), collapse = ", "), "\n")
cat("Columns in val but not in train_smote:", 
    paste(setdiff(names(val), names(train_smote)), collapse = ", "), "\n")

# 3. Export
write.csv(train_smote, "data/train_final.csv", row.names = FALSE)
write.csv(val, "data/validation_final.csv", row.names = FALSE)
write.csv(test, "data/test_final.csv", row.names = FALSE)