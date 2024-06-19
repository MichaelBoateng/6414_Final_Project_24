
library(tidyverse)  # For data manipulation and visualization
library(caret)      # For creating data partitions
library(broom)      # For tidying up model outputs
library(plotly)     # For interactive visualizations
library(dplyr)      # For data manipulation
library(tidyr)      # For data manipulation
library(ggplot2)    # For data visualization
library(corrplot)   # For correlation plot

library(randomForest)
library(glmnet)   # For Ridge, Lasso, and Elastic Net
library(gbm)      # For Gradient Boosting
library(e1071)    # For SVM
library(neuralnet) # For Neural Networks

library(readxl)
df <- read_excel("powerconsumption.xlsx")

head(df)

#3. Data Inspection and Cleaning:

# Data shape
dim(df)

# Calculate mean of PowerConsumption_Zone1, PowerConsumption_Zone2, and PowerConsumption_Zone3
df <- df %>% mutate(PowerConsumption_Zone = rowMeans(select(., starts_with('PowerConsumption_Zone'))))
head(df)

# Check for missing values
colSums(is.na(df))

# Check for duplicate rows
sum(duplicated(df))

# Summary statistics
summary(df)







#Data Transformation - Standardization

# Define numerical columns
numerical_columns <- c('Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 
                       'DiffuseFlows', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 
                       'PowerConsumption_Zone3', 'PowerConsumption_Zone')

# Standardize the data
preProcValues <- preProcess(df[numerical_columns], method = c("center", "scale"))
df_scaled <- predict(preProcValues, df[numerical_columns])

# Display the standardized data
head(df_scaled)

# Normalize the data
preProcValues_norm <- preProcess(df[numerical_columns], method = c("range"))
df_normalized <- predict(preProcValues_norm, df[numerical_columns])

# Display the normalized data
head(df_normalized)

#Data Preparation and Train-Test Split

# Define numerical columns
numerical_columns <- c('Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 
                       'DiffuseFlows', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 
                       'PowerConsumption_Zone3', 'PowerConsumption_Zone')

# Standardize the data
preProcValues <- preProcess(df[numerical_columns], method = c("center", "scale"))
df_scaled <- predict(preProcValues, df[numerical_columns])

# Define X and y
X <- df_scaled
y <- df$PowerConsumption_Zone

# Train-Test Split
set.seed(42)
trainIndex <- createDataPartition(y, p = .8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

#Model Training and Evaluation - Linear Regression

# Model Training - Linear Regression
linear_model <- lm(y_train ~ ., data = as.data.frame(cbind(y_train, X_train)))

# Model Prediction
y_pred_lr <- predict(linear_model, as.data.frame(X_test))

# Model Evaluation
mse_lr <- mean((y_test - y_pred_lr)^2)
print(paste('Mean Squared Error for Linear Regression:', mse_lr))

# Create a data frame for plotting
plot_data_lr <- data.frame(
  Actual = y_test,
  Predicted = y_pred_lr,
  Residuals = y_test - y_pred_lr
)

# Plot the predicted values vs the actual values with larger, centered labels
ggplot(plot_data_lr, aes(x = Actual, y = Predicted)) +
  geom_point() +  # Scatter plot
  geom_abline(slope = 1, intercept = 0, color = 'red') +  # 45-degree line
  labs(title = 'Actual vs Linear Regression Predicted Values', x = 'Actual Values', y = 'Predicted Values') +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 30, hjust = 0.5),  # Title text size and alignment
    axis.title.x = element_text(size = 20, hjust = 0.5),  # X-axis title text size and alignment
    axis.title.y = element_text(size = 20, hjust = 0.5),  # Y-axis title text size and alignment
    axis.text.x = element_text(size = 12, hjust = 0.5),  # X-axis text size and alignment
    axis.text.y = element_text(size = 12, hjust = 0.5)   # Y-axis text size and alignment
  )
ggsave("images/actual_vs_linear.png")

#Model Training and Evaluation - Random Forest

# Model Training - Random Forest
rf_model <- randomForest(X_train, y_train, ntree = 2000, random_state = 42)

# Model Prediction
y_pred_rf <- predict(rf_model, X_test)

# Model Evaluation
mse_rf <- mean((y_test - y_pred_rf)^2)
print(paste('Mean Squared Error for Random Forest:', mse_rf))

# Create a data frame for plotting
plot_data_rf <- data.frame(
  Actual = y_test,
  Predicted = y_pred_rf
)

# Plot the predicted values vs the actual values with larger, centered labels
ggplot(plot_data_rf, aes(x = Actual, y = Predicted)) +
  geom_point() +  # Scatter plot
  geom_abline(slope = 1, intercept = 0, color = 'blue') +  # 45-degree line
  labs(title = 'Actual vs Random Forest Regression Predicted Values', x = 'Actual Values', y = 'Predicted Values') +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 30, hjust = 0.5),  # Title text size and alignment
    axis.title.x = element_text(size = 20, hjust = 0.5),  # X-axis title text size and alignment
    axis.title.y = element_text(size = 20, hjust = 0.5),  # Y-axis title text size and alignment
    axis.text.x = element_text(size = 12, hjust = 0.5),  # X-axis text size and alignment
    axis.text.y = element_text(size = 12, hjust = 0.5)   # Y-axis text size and alignment
)
ggsave("images/actual_vs_random.png")

# Ridge Regression
ridge_model <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0)  # alpha = 0 for Ridge
y_pred_ridge <- predict(ridge_model, s = ridge_model$lambda.min, newx = as.matrix(X_test))
mse_ridge <- mean((y_test - y_pred_ridge)^2)
print(paste('Mean Squared Error for Ridge Regression:', mse_ridge))

# Lasso Regression
lasso_model <- cv.glmnet(as.matrix(X_train), y_train, alpha = 1)  # alpha = 1 for Lasso
y_pred_lasso <- predict(lasso_model, s = lasso_model$lambda.min, newx = as.matrix(X_test))
mse_lasso <- mean((y_test - y_pred_lasso)^2)
print(paste('Mean Squared Error for Lasso Regression:', mse_lasso))

# Elastic Net
elastic_model <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0.5)  # alpha = 0.5 for Elastic Net
y_pred_elastic <- predict(elastic_model, s = elastic_model$lambda.min, newx = as.matrix(X_test))
mse_elastic <- mean((y_test - y_pred_elastic)^2)
print(paste('Mean Squared Error for Elastic Net:', mse_elastic))

# Comparing Mean Squared Errors
mse_results <- data.frame(
  Model = c("Linear Regression", "Random Forest", "Ridge Regression", "Lasso Regression", 
            "Elastic Net"),
  MSE = c(mse_lr, mse_rf, mse_ridge, mse_lasso, mse_elastic)
)
print(mse_results)


# Creating a Result DataFrame
result_df <- data.frame(
  Actual = y_test,
  Predicted_LR = y_pred_lr,
  Predicted_RF = y_pred_rf
)
result_df <- result_df %>%
  mutate(
    LR_Error = abs(Actual - Predicted_LR),
    RF_Error = abs(Actual - Predicted_RF))
head(result_df)


#2. Residual Plots

# Calculate residuals
result_df$Residuals_LR <- result_df$Actual - result_df$Predicted_LR
result_df$Residuals_RF <- result_df$Actual - result_df$Predicted_RF

# Plot residuals for linear regression
ggplot(result_df, aes(x = Actual, y = Residuals_LR)) +
  geom_point(color = 'blue') +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  labs(title = "Residual Plot for Linear Regression", x = "Actual Values", y = "Residuals")+
  theme_minimal() +
  theme(
    plot.title = element_text(size = 30, hjust = 0.5),  # Title text size and alignment
    axis.title.x = element_text(size = 20, hjust = 0.5),  # X-axis title text size and alignment
    axis.title.y = element_text(size = 20, hjust = 0.5),  # Y-axis title text size and alignment
    axis.text.x = element_text(size = 12, hjust = 0.5),  # X-axis text size and alignment
    axis.text.y = element_text(size = 12, hjust = 0.5)   # Y-axis text size and alignment
)
ggsave("images/variance_linear.png")

# Plot residuals for random Forest
ggplot(result_df, aes(x = Actual, y = Residuals_RF)) +
  geom_point(color = 'blue') +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed') +
  theme_minimal() +
  labs(title = "Residual Plot for Random Forest", x = "Actual Values", y = "Residuals")+
  theme_minimal() +
  theme(
    plot.title = element_text(size = 30, hjust = 0.5),  # Title text size and alignment
    axis.title.x = element_text(size = 20, hjust = 0.5),  # X-axis title text size and alignment
    axis.title.y = element_text(size = 20, hjust = 0.5),  # Y-axis title text size and alignment
    axis.text.x = element_text(size = 12, hjust = 0.5),  # X-axis text size and alignment
    axis.text.y = element_text(size = 12, hjust = 0.5)   # Y-axis text size and alignment
)
ggsave("images/variance_random.png")

#3. Distribution of Residuals
# Calculate residuals for random forest
result_df$Residuals_RF <- result_df$Actual - result_df$Predicted_RF

# Combine residuals into long format for plotting
residuals_long <- result_df %>%
  gather(key = "Model", value = "Residuals", Residuals_LR, Residuals_RF)

# Plot the distribution of residuals
ggplot(residuals_long, aes(x = Residuals, fill = Model)) +
  geom_histogram(bins = 30, alpha = 0.7, position = 'identity') +
  facet_wrap(~ Model, scales = 'free') +
  theme_minimal() +
  labs(title = "Distribution of Residuals for Linear Regression and Random Forest", x = "Residuals", y = "Frequency") +
  theme(
    plot.title = element_text(size = 20, hjust = 0.5),  # Title text size and alignment
    axis.title.x = element_text(size = 15, hjust = 0.5),  # X-axis title text size and alignment
    axis.title.y = element_text(size = 15, hjust = 0.5),  # Y-axis title text size and alignment
    axis.text.x = element_text(size = 12, hjust = 0.5),  # X-axis text size and alignment
    axis.text.y = element_text(size = 12, hjust = 0.5)   # Y-axis text size and alignment
)

ggsave("images/residuals_distribution.png", width = 12, height = 6)