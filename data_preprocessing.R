library(tidyverse)
library(caret)
library(fastDummies)

bank <- read.csv("data/bank-additional-full.csv", sep = ";")

cat("Initial dimensions:", dim(bank)[1], "rows ×", dim(bank)[2], "columns\n\n")

# Check & remove duplicate rows
dup_count <- sum(duplicated(bank))
if (dup_count > 0) {
  cat("Duplicate rows detected:", dup_count, "— removing...\n")
  bank <- bank %>% distinct()
  cat("Duplicates removed. New dimensions:", dim(bank)[1], "rows ×", dim(bank)[2], "columns\n\n")
} else {
  cat("No duplicate rows detected.\n\n")
}

# Check & remove constant columns
constant_cols <- sapply(bank, function(col) length(unique(col)) == 1)
if (any(constant_cols)) {
  cat("Constant columns detected and removed:\n")
  print(names(bank)[constant_cols])
  bank <- bank[, !constant_cols]
  cat("Remaining columns:", ncol(bank), "\n\n")
} else {
  cat("No constant columns detected.\n\n")
}

# Check missing values
check_missing_vals <- function(data) {
  na_summary <- colSums(is.na(data))
  na_present <- na_summary[na_summary > 0]
  if (length(na_present) > 0) {
    cat("Columns with missing values detected:\n")
    print(na_present)
    cat("Recommendation: impute or remove these records before modeling.\n\n")
  } else {
    cat("No missing values detected.\n\n")
  }
}
check_missing_vals(bank)

# Stratified train / validation / test Split
set.seed(1234)

trainIndex <- createDataPartition(bank$y, p = 0.70, list = FALSE)
trainData  <- bank[trainIndex, ]
tempData   <- bank[-trainIndex, ]

valIndex <- createDataPartition(tempData$y, p = 0.50, list = FALSE)
valData  <- tempData[valIndex, ]
testData <- tempData[-valIndex, ]

cat("Train size:", nrow(trainData), round(mean(trainData$y == "yes") * 100, 2),
    "\nValidation size:", nrow(valData), round(mean(valData$y == "yes") * 100, 2),
    "\nTest size:", nrow(testData), round(mean(testData$y == "yes") * 100, 2))

# Define preprocessing function
preprocess_bank <- function(df, fit_obj = NULL) {
  d <- df
  
  # Binary encoding
  d$y <- ifelse(d$y == "yes", TRUE, FALSE)
  d$default <- ifelse(d$default == "yes", 1, 0)
  d$housing <- ifelse(d$housing == "yes", 1, 0)
  d$loan    <- ifelse(d$loan == "yes", 1, 0)
  
  # Ordinal encoding for education
  edu_levels <- c("illiterate", "basic.4y", "basic.6y", "basic.9y",
                  "high.school", "professional.course", "university.degree", "unknown")
  d$education <- factor(d$education, levels = edu_levels, ordered = TRUE)
  d$education_num <- as.numeric(d$education)
  d <- d[, !(names(d) %in% "education")]
  
  
  # Month / Quarter
  month_map <- setNames(1:12,
                        c("jan","feb","mar","apr","may","jun","jul",
                          "aug","sep","oct","nov","dec"))
  d$month_num <- month_map[as.character(d$month)]
  d$quarter <- ceiling(d$month_num / 3)
  
  # Age group
  d$age_group <- cut(d$age,
                     breaks = c(-Inf, 30, 40, 50, 60, Inf),
                     labels = c("18-30","31-40","41-50","51-60","60+"),
                     right = FALSE)
  
  # Campaign intensity
  d$call_effort <- with(d, campaign / (1 + previous))
  
  # Economic context composite
  econ_vars <- c("emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed")
  d$econ_index <- rowMeans(scale(d[, econ_vars]))
  
  # Dummy encoding
  d <- fastDummies::dummy_cols(d,
                               select_columns = c("job","marital","contact",
                                                  "month","day_of_week","poutcome","age_group"),
                               remove_first_dummy = TRUE,
                               remove_selected_columns = TRUE)
  
  # Scaling (fit on training data only)
  num_vars <- sapply(d, is.numeric)
  if (is.null(fit_obj)) {
    fit_obj <- preProcess(d[, num_vars], method = c("center", "scale"))
  }
  d[num_vars] <- predict(fit_obj, d[, num_vars])
  
  list(data = d, scaler = fit_obj)
}

# Apply preprocessing
prep_train <- preprocess_bank(trainData)
train_prep <- prep_train$data
scaler_fit <- prep_train$scaler

val_prep  <- preprocess_bank(valData, fit_obj = scaler_fit)$data
test_prep <- preprocess_bank(testData, fit_obj = scaler_fit)$data

# Re-check missing values
check_missing_vals(train_prep)
check_missing_vals(val_prep)
check_missing_vals(test_prep)

# Convert target y from logical to numeric (1/0)
train_prep$y <- ifelse(train_prep$y == TRUE, 1, 0)
val_prep$y   <- ifelse(val_prep$y == TRUE, 1, 0)
test_prep$y  <- ifelse(test_prep$y == TRUE, 1, 0)

# Export prepared datasets
write.csv(train_prep, "data/train_prepared.csv", row.names = FALSE)
write.csv(val_prep,  "data/validation_prepared.csv", row.names = FALSE)
write.csv(test_prep, "data/test_prepared.csv", row.names = FALSE)