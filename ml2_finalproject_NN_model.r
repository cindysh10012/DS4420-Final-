# Load libraries
library(neuralnet)
library(ggplot2)
library(scales)

# Read and prepare data
df <- read.csv("Apple Financial Stamt Data 24_15.csv")
df$Quarter <- as.Date(df$Quarter, format="%m/%d/%Y")
df <- df[order(df$Quarter), ]

# Normalize the revenue for better neural net performance
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
denormalize <- function(x_norm, orig) {
  x_norm * (max(orig) - min(orig)) + min(orig)
}

df$Revenue_norm <- normalize(df$Revenue)

# Create training and test sets
cutoff <- floor(nrow(df) * 0.8)
train <- df[1:cutoff, ]
test <- df[(cutoff + 1):nrow(df), ]


# Create 4 lag variables in training data
# Lag1: previous observation
train$Lag1 <- c(NA, head(train$Revenue_norm, -1))
# Lag2: observation from 2 periods ago
train$Lag2 <- c(NA, NA, head(train$Revenue_norm, -2))
# Lag3: observation from 3 periods ago
train$Lag3 <- c(NA, NA, NA, head(train$Revenue_norm, -3))
# Lag4: observation from 4 periods ago
train$Lag4 <- c(NA, NA, NA, NA, head(train$Revenue_norm, -4))

# Remove the first four rows that now contain NA values
train <- train[-(1:4), ]

# Train neural network model using 4 lags
nn <- neuralnet(Revenue_norm ~ Lag1 + Lag2 + Lag3 + Lag4, 
                data = train, 
                hidden = c(4, 3), 
                linear.output = TRUE)

# For the first test row, we need the last 4 revenue values from the training set.
# Then we append the remaining test values lagged accordingly.
test$Lag1 <- c(tail(train$Revenue_norm, 1), head(test$Revenue_norm, -1))
test$Lag2 <- c(tail(train$Revenue_norm, 2), head(test$Revenue_norm, -2))
test$Lag3 <- c(tail(train$Revenue_norm, 3), head(test$Revenue_norm, -3))
test$Lag4 <- c(tail(train$Revenue_norm, 4), head(test$Revenue_norm, -4))

# Create a data frame containing all 4 lag variables for the test input.
test_input <- data.frame(
  Lag1 = test$Lag1,
  Lag2 = test$Lag2,
  Lag3 = test$Lag3,
  Lag4 = test$Lag4
)

# Generate predictions on the test set
test_pred <- compute(nn, test_input)
test$Predicted_norm <- test_pred$net.result

# Convert the normalized predictions back to the original scale
test$Predicted <- denormalize(test$Predicted_norm, df$Revenue)

# Combine training and test actual + predicted values for plotting
# For the training set, denormalize the actual values for consistency
train$Predicted <- denormalize(train$Revenue_norm, df$Revenue)

train_plot <- data.frame(Quarter = train$Quarter, Revenue = train$Revenue)
test_plot <- data.frame(
  Quarter = test$Quarter, 
  Actual = test$Revenue, 
  Predicted = denormalize(test$Predicted_norm, df$Revenue)
)

# Plot
ggplot() +
  # Training Revenue
  geom_line(data=train_plot, aes(x=Quarter, y=Revenue, color="Training Revenue"), linewidth=1) +
  
  # Actual Revenue in Test Period (now includes last training point)
  geom_line(data=test_plot, aes(x=Quarter, y=Actual, color="Actual Revenue"), linewidth=1) +
  
  # Predicted Revenue in Test Period
  geom_line(data=test_plot[-1, ], aes(x=Quarter, y=Predicted, color="Predicted Revenue"), 
            linewidth=1, linetype="dashed") +  # exclude patched row
  
  # Title and labels
  labs(title="Forecasting Quarterly Revenue with Neural Network",
       x="Quarter", y="Revenue", color="Legend") +
  
  # Axis formatting
  scale_y_continuous(labels=scales::comma) +
  scale_color_manual(values = c("Training Revenue" = "steelblue",
                                "Actual Revenue" = "orange",
                                "Predicted Revenue" = "darkgreen")) +
  
  theme_minimal(base_size = 13) +
  theme(legend.position = "top",
        plot.title = element_text(face="bold", size=15, hjust = 0.5))

# Remove NA rows before evaluation and plotting
test_plot_clean <- na.omit(test_plot)

# Evaluation metrics
mse <- mean((test_plot_clean$Actual - test_plot_clean$Predicted)^2)
mae <- mean(abs(test_plot_clean$Actual - test_plot_clean$Predicted))
mape <- mean(abs((test_plot_clean$Actual - test_plot_clean$Predicted) / test_plot_clean$Actual)) * 100

cat("MSE: ", round(mse, 2), "\n")
cat("MAE: ", round(mae, 2), "\n")
cat("MAPE: ", round(mape, 2), "%\n")

# Predicted vs. Actual Scatter Plot
ggplot(test_plot_clean, aes(x=Predicted, y=Actual)) +
  geom_point(size=3, color="steelblue") +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="red", linewidth=1) +
  labs(title="Predicted vs. Actual Revenue",
       x="Predicted Revenue",
       y="Actual Revenue") +
  theme_minimal(base_size = 13)
