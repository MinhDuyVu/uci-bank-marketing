library(e1071)
library(pROC)
library(tidyverse)
library(caret)

set.seed(1)

# 1. Load Data

train_data <- read.csv("train_final.csv")
test_data  <- read.csv("test_final.csv")

# Ensure target is factor
train_data$y <- as.factor(train_data$y)
test_data$y  <- as.factor(test_data$y)

# 2. Build a subsample for fast tuning

sample_size <- min(4000, nrow(train_data))   
sample_idx <- sample(nrow(train_data), sample_size)
tune_train <- train_data[sample_idx, ]

cat("Tuning on subsample of", nrow(tune_train), "rows\n")

# 3. Tune SVM (small grid)

tuned.svm <- tune(svm,
                  y ~ .,
                  data = tune_train,
                  kernel = "radial",
                  ranges = list(cost = c(1, 10),
                                gamma = c(0.01, 0.1)),
                  scale = TRUE,
                  probability = TRUE)

cat("Tuning summary:\n")
summary(tuned.svm)

# Extract best model parameters
best.params <- tuned.svm$best.parameters
cat("Best parameters:\n")
print(best.params)

# 4. Retrain best model on FULL training data

best.svm <- svm(y ~ .,
                data = train_data,
                kernel = "radial",
                cost = best.params$cost,
                gamma = best.params$gamma,
                scale = TRUE,
                probability = TRUE)

cat("Retraining completed on full training data.\n")

# 5. Predictions on Test Set

svm.pred <- predict(best.svm, test_data %>% select(-y))
svm.prob <- attr(predict(best.svm, test_data %>% select(-y), probability = TRUE),
                 "probabilities")[, 2]

# 6. Evaluate Overall Metrics

svm.accuracy <- mean(svm.pred == test_data$y)
cat("\nSVM Accuracy:", round(svm.accuracy, 4), "\n")

conf_matrix <- confusionMatrix(svm.pred, test_data$y, positive = "1")
print(conf_matrix)

precision <- conf_matrix$byClass["Precision"]
recall    <- conf_matrix$byClass["Recall"]
f1        <- conf_matrix$byClass["F1"]

cat("\nPrecision:", round(precision, 4),
    "\nRecall:   ", round(recall, 4),
    "\nF1-Score: ", round(f1, 4), "\n")

roc_obj <- roc(as.numeric(test_data$y), as.numeric(svm.prob))
auc_value <- auc(roc_obj)
cat("AUC-ROC:", round(auc_value, 4), "\n")

plot(roc_obj, col = "blue", lwd = 2, main = "SVM ROC Curve (Option B Original)")
abline(a = 0, b = 1, lty = 2, col = "red")

# 7. Evaluate by Segments (Age groups)

segment_cols <- c("age_group_31.40", "age_group_41.50", "age_group_51.60")

segment_results <- data.frame(Segment = character(),
                              Accuracy = numeric(),
                              Precision = numeric(),
                              Recall = numeric(),
                              F1 = numeric(),
                              AUC = numeric(),
                              stringsAsFactors = FALSE)

for (col in segment_cols) {
  if (!col %in% names(test_data)) next
  seg <- subset(test_data, get(col) > 0)
  if (nrow(seg) == 0) next
  
  seg_pred <- predict(best.svm, seg %>% select(-y))
  seg_prob <- attr(predict(best.svm, seg %>% select(-y), probability = TRUE), "probabilities")[, 2]
  
  cm_seg <- confusionMatrix(seg_pred, seg$y, positive = "1")
  roc_seg <- roc(as.numeric(seg$y), as.numeric(seg_prob))
  
  segment_results <- rbind(segment_results, data.frame(
    Segment = col,
    Accuracy = as.numeric(cm_seg$overall["Accuracy"]),
    Precision = as.numeric(cm_seg$byClass["Precision"]),
    Recall = as.numeric(cm_seg$byClass["Recall"]),
    F1 = as.numeric(cm_seg$byClass["F1"]),
    AUC = as.numeric(auc(roc_seg))
  ))
}

cat("\nPerformance by Age Segments:\n")
print(segment_results)

# 8. Economic Condition Evaluation

if ("emp.var.rate" %in% names(test_data)) {
  median_emp <- median(test_data$emp.var.rate, na.rm = TRUE)
  low_emp <- subset(test_data, emp.var.rate < median_emp)
  high_emp <- subset(test_data, emp.var.rate >= median_emp)
  
  for (segment_name in c("Low Employment", "High Employment")) {
    seg <- if (segment_name == "Low Employment") low_emp else high_emp
    if (nrow(seg) == 0) next
    seg_pred <- predict(best.svm, seg %>% select(-y))
    seg_prob <- attr(predict(best.svm, seg %>% select(-y), probability = TRUE), "probabilities")[, 2]
    
    cm_seg <- confusionMatrix(seg_pred, seg$y, positive = "1")
    roc_seg <- roc(as.numeric(seg$y), as.numeric(seg_prob))
    
    cat("\n--- Segment:", segment_name, "---\n",
        "Accuracy:", round(as.numeric(cm_seg$overall["Accuracy"]), 4), "\n",
        "Precision:", round(as.numeric(cm_seg$byClass["Precision"]), 4), "\n",
        "Recall:", round(as.numeric(cm_seg$byClass["Recall"]), 4), "\n",
        "F1-Score:", round(as.numeric(cm_seg$byClass["F1"]), 4), "\n",
        "AUC-ROC:", round(as.numeric(auc(roc_seg)), 4), "\n")
  }
} 

