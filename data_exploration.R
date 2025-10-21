library(tidyverse)
library(skimr)
library(ggplot2)
library(GGally)
library(corrplot)
library(reshape2)
library(scales)
library(ggthemes)
library(psych)
library(RColorBrewer)
library(vcd)


# 1. Load data
bank <- read.csv("data/bank-additional-full.csv", sep = ";")
glimpse(bank)

# 2. Structural Summary
cat("\nDataset dimensions:\n")
print(dim(bank))

cat("\nVariable types:\n")
print(sapply(bank, class))

skim(bank)  # Summary statistics

# 3. Target Variable Analysis
table(bank$y)
prop.table(table(bank$y)) * 100

ggplot(bank, aes(x = y, fill = y)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Class Distribution: Term Deposit Subscription",
       x = "Subscription", y = "Count") +
  theme_minimal()

# 4. Customer Demographics
# Age
ggplot(bank, aes(x = y, y = age, fill = y)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Subscription Status") +
  theme_minimal()

# Job type
job_summary <- bank %>%
  group_by(job) %>%
  summarise(SubscriptionRate = mean(y == "yes") * 100,
            Count = n()) %>%
  arrange(desc(SubscriptionRate))

ggplot(job_summary, aes(x = reorder(job, SubscriptionRate), y = SubscriptionRate)) +
  geom_col(fill = "#0073C2FF") +
  coord_flip() +
  labs(title = "Subscription Rate by Job Type", x = "Job", y = "Subscription Rate (%)") +
  theme_minimal()

# Education
edu_summary <- bank %>%
  group_by(education) %>%
  summarise(SubscriptionRate = mean(y == "yes") * 100)

ggplot(edu_summary, aes(x = reorder(education, SubscriptionRate), y = SubscriptionRate)) +
  geom_col(fill = "#00A087FF") +
  coord_flip() +
  labs(title = "Subscription Rate by Education Level", y = "Subscription Rate (%)") +
  theme_minimal()

# Default status
ggplot(bank, aes(x = default, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Credit Default vs Term Deposit Subscription",
       y = "Proportion of Clients") +
  theme_minimal()

# Loan status
loan_table <- bank %>%
  group_by(housing, y) %>%
  summarise(count = n()) %>%
  mutate(pct = count / sum(count) * 100)

ggplot(loan_table, aes(x = housing, y = pct, fill = y)) +
  geom_col(position = "dodge") +
  labs(title = "Housing Loan vs Subscription", y = "Percentage") +
  theme_minimal()

# 5. Campaign Characteristics
# Contact type
ggplot(bank, aes(x = contact, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Contact Type", y = "Proportion") +
  theme_minimal()

# Month
bank$month <- factor(bank$month, levels = month.abb) # ensure chronological order if coded as Jan-Dec
ggplot(bank, aes(x = month, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Month", y = "Proportion") +
  theme_minimal()

# Call duration
ggplot(bank, aes(x = duration, fill = y)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  xlim(0, quantile(bank$duration, 0.95)) +
  labs(title = "Call Duration Distribution", x = "Duration (seconds)", y = "Count") +
  theme_minimal()

# Number of contacts
ggplot(bank, aes(x = campaign, fill = y)) +
  geom_histogram(binwidth = 1, position = "dodge") +
  xlim(0, 30) +
  labs(title = "Number of Contacts vs Subscription", x = "Contacts in Campaign") +
  theme_minimal()


# 7. Previous Campaign History
ggplot(bank, aes(x = pdays, fill = y)) +
  geom_histogram(bins = 40, position = "identity", alpha = 0.6) +
  xlim(0, 1000) +
  labs(title = "Days Since Last Contact vs Subscription") +
  theme_minimal()

ggplot(bank, aes(x = previous, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Previous Contacts vs Subscription") +
  theme_minimal()

ggplot(bank, aes(x = poutcome, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Previous Campaign Outcome vs Current Subscription", y = "Proportion") +
  theme_minimal()

# 8. Economic Indicators
econ_vars <- c("emp.var.rate", "cons.price.idx", "cons.conf.idx",
               "euribor3m", "nr.employed")

econ_summary <- bank %>%
  group_by(y) %>%
  summarise(across(all_of(econ_vars), mean))

print(econ_summary)

# Euribor vs subscription
ggplot(bank, aes(x = euribor3m, fill = y)) +
  geom_density(alpha = 0.5) +
  labs(title = "Euribor 3-Month Rate vs Subscription Probability") +
  theme_minimal()

# 9. Correlation Analysis=
num_vars <- bank %>%
  select(where(is.numeric))

cor_matrix <- cor(num_vars)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.6, title = "Correlation Heatmap of Numeric Features")

# 10. Association Tests (Categorical)
categoricals <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome")

assoc_results <- lapply(categoricals, function(var) {
  tbl <- table(bank[[var]], bank$y)
  if (any(rowSums(tbl) == 0) || any(colSums(tbl) == 0)) {
    return(data.frame(Variable = var,
                      ChiSq = NA,
                      p_value = NA,
                      Note = "Skipped (zero count in a row/column)"))
  }
  
  test <- suppressWarnings(chisq.test(tbl))
  if (is.null(test)) {
    data.frame(Variable = var,
               ChiSq = NA,
               p_value = NA,
               Note = "Error in test")
  } else {
    data.frame(Variable = var,
               ChiSq = round(test$statistic, 3),
               p_value = signif(test$p.value, 4),
               Note = "OK")
  }
})
assoc_df <- as.data.frame(do.call(rbind, assoc_results))
print(assoc_df)

# 11. Export Summary Outputs
write.csv(job_summary, "data/job_subscription_summary.csv", row.names = FALSE)
write.csv(edu_summary, "data/education_subscription_summary.csv", row.names = FALSE)
write.csv(assoc_df, "data/categorical_association_tests.csv", row.names = FALSE)
