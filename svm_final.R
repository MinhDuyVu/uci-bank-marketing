library(pROC) 
library(e1071)
library(tidyverse)
library(caret)

train_data <- read.csv("train_final.csv")
test_data <- read.csv("test_final.csv")

#Ensure target is a factor
train_data$y <- as.factor(train_data$y)
test_data$y <- as.factor(test_data$y)

#Tune and Train SVM Model
class_counts <- table(train_data$y)

class_weights <- as.list(class_counts["0"] / class_counts)
names(class_weights) <- names(class_counts)
print(class_weights)

tuned.svm <- tune(svm,
                  y ~ .,
                 data = train_data,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.01, 0.1, 1)),
                 class.weights = class_weights,
                 scale = TRUE)


##Select best model
summary(tuned.svm)
best.svm <- tuned.svm$best.model

#Prediction and Probability
svm.pred <- predict(best.svm, test_data[, -which(names(test_data) == "y")])
svm.prob <- attr(predict(best.svm, test_data[, -which(names(test_data) == "y")],
                         probability = TRUE), "probabilities")[, 2]

#Test Accuracy
svm.accuracy <- mean(svm.pred == test_data$y)
print(paste("SVM Accuracy:", round(svm.accuracy, 4)))

#Show Precision, Recall and F-1 Score
conf_matrix <- confusionMatrix(svm.pred, test_data$y, positive = "1")
print(conf_matrix)

precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1 <- conf_matrix$byClass["F1"]

cat("\nPrecision", round(precision, 4),
    "\nRecall:", round(recall, 4),
    "\nF!-Score:", round(f1, 4), "\n")

#AUC-ROC Curve
roc_obj <- roc(as.numeric(test_data$y), as.numeric(svm.prob))
auc_value <- auc(roc_obj)
cat("AUC-ROC:", round(auc_value, 4), "\n")

plot(rob_obj, col = "blue", lwd = 2, main = "SVM ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "red")

##Evaluate Segments
#Segment - Age Groups
segment_cols <- c("age_group_31.40", "age_group_41.50", "age_group_51.60")

segment_results <- data.frame(Segment = character(),
                              Accuracy = numeric(),
                              Precision = numeric(),
                              F1 = numeric(),
                              AUC = numeric())

for (col in segment_cols) {
  seg <- subset(test_data, get(col) > 0)
  if (nrow(seg) > 0) {
    seg_pred <- predict(best.svm, seg[, -which(names(seg) == "y")])
    seg_prob <- attr(predict(best.svm, seg[, -which(names(seg) == "y")],
                             probability = TRUE), "probabilities")[, 2]
    cm_seg <- confusionMatrix(seg_pred, seg$y, positive = "1")
    roc_seg <- roc(as.numeric(seg$y), as.numeric(seg_prob))
    
    segment_results <- rbind(segment_results, data.frame(
      Segment = col,
      Accuracy = cm_seg$overall["Accuracy"],
      Precision = cm_seg$byClass["Precision"],
      Recall = cm_seg$byClass["Recall"],
      F1 = cm_seg$byClass["F1"],
      AUC = auc(roc_seg)
    ))
  }
}

cat("\nPerformance by Age\n")
print(segment_results)

#Segment - Economic Conditions
median_emp <- median(test_data$emp.var.rate)

low_emp <- subset(test_data, emp.var.rate < median_emp)
high_emp <- subset(test_data, emp.var.rate >= median_emp)

for (segment_name in c("Low Employment", "High Employment")) {
  seg <- if (segment_name == "Low Employment") low_emp else high_emp
  seg_pred <- predict(best.svm, seg[, -which(names(seg) == "y")])
  seg_prob <- attr(predict(best.svm, seg[, -which(names(seg) == "y")],
                           probability = TRUE), "probabilities")[, 2]
  
  cm_seg <- confusionMatrix(seg_pred, seg$y, positive = "1")
  roc_seg <- roc(as.numeric(seg$y), as.numeric(seg_prob))
  
  cat("\nEconomic Conditions Segment:", segment_name, "---\n",
      "Accuracy:", round(cm_seg$overall["Accuracy"], 4), "\n",
      "Precision:", round(cm_seg$byClass["Precision"], 4), "\n",
      "Recall:", round(cm_seg$byClass["Recall"], 4), "\n",
      "F1-Score:", round(cm_seg$byClass["F1"], 4), "\n",
      "AUC-ROC:", round(auc(roc_seg), 4), "\n")

