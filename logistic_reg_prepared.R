library(tidyverse)
library(caret)
library(pROC)

set.seed(42)

train <- read.csv("data/train_prepared.csv")
val   <- read.csv("data/validation_prepared.csv")
test  <- read.csv("data/test_prepared.csv")

# 1. Ensure target is factor for caret
train$y <- factor(ifelse(train$y == TRUE, "yes", "no"),
                  levels = c("no", "yes"))
val$y   <- factor(ifelse(val$y == TRUE, "yes", "no"),
                  levels = c("no", "yes"))

# 2. Cross-Validation with SMOTE sampling
cv_ctrl <- trainControl(
  method = "cv",
  number = 5,
  sampling = "smote",
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

# 3. Train Logistic Regression (baseline)
model_logit <- train(
  y ~ .,
  data = train,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = cv_ctrl
)

cat("\nCross-validated AUC (mean):",
    round(mean(model_logit$resample$ROC), 3), "\n")

# 4. Evaluate on Validation Set
pred_val_prob <- predict(model_logit, val, type = "prob")[, "yes"]
pred_val <- ifelse(pred_val_prob > 0.5, "yes", "no")

cm <- confusionMatrix(
  factor(pred_val, levels = c("no", "yes")),
  val$y,
  positive = "yes"
)
print(cm)

auc_val <- roc(response = val$y, predictor = pred_val_prob)$auc
cat("\nValidation AUC:", round(auc_val, 3), "\n")

# 5. Business Cost Interpretation
FP_cost <- 5
FN_loss <- 100
FP <- cm$table["yes", "no"]
FN <- cm$table["no", "yes"]
expected_loss <- FP_cost * FP + FN_loss * FN
cat("\nEstimated business loss (illustrative): $", expected_loss, "\n")

# 6. Report Metrics
cat("\nPerformance Metrics:\n")
cat("Accuracy :", round(cm$overall['Accuracy'], 3), "\n")
cat("Precision:", round(cm$byClass['Precision'], 3), "\n")
cat("Recall   :", round(cm$byClass['Recall'], 3), "\n")
cat("F1 Score :", round(cm$byClass['F1'], 3), "\n")
cat("AUC      :", round(auc_val, 3), "\n")