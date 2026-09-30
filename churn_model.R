# Telco Customer Churn: preprocessing, visualisation and predictive modelling
# Author: Gia Huy Dang
# Run from the repository root: Rscript churn_model.R

# Load required libraries for preprocessing, analysis, and visualization
library(dplyr)
library(ggplot2)    
library(VIM)         
library(mice)        
library(moments)    
library(scales)      
library(patchwork)   
library(knitr)       
library(corrplot)    
library(reshape2)
library(caret)
library(ggcorrplot)
library(randomForest)
library(viridis)
library(smotefamily)          
library(pROC)
library(ggthemes)
library(glmnet)
library(xgboost)

# Load the Kaggle Telco Customer Churn dataset (see data/README.md)
data <- read.csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
dir.create("outputs", showWarnings = FALSE)

# When run with Rscript, the default PDF device does not know "Arial"; map it to Helvetica
if (!"Arial" %in% names(pdfFonts())) pdfFonts(Arial = pdfFonts()$Helvetica)

################################################################################
##################### Task 2: Data pre-processing ##############################
################################################################################

############# Step 1: Initial Data Inspection and Error Correction ################
# Objective: Investigate potential data quality issues (missing values, categorical errors, imbalance, outliers, variable types) and correct errors where applicable.

##### Step 1.1: Dataset Overview #####
cat("Step 1.1: Dataset Overview\n")
cat("----------------------------\n")

# Ensure TotalCharges is numeric, handling potential blanks/non-numeric values
data$TotalCharges <- as.numeric(as.character(data$TotalCharges))  

# Check dataset dimensions
cat("Dataset dimensions (rows, columns): ", dim(data), "\n")  

# Review structure of the dataset
cat("\nDataset structure:\n")
str(data)  # Displays data types and sample values for each column

# Summarize variable types
cat("\nVariable type summary:\n")
var_types <- data.frame(
  Column = names(data),
  Type = sapply(data, class)
)
kable(var_types, caption = "Variable Types")

##### Step 1.2: Missing Value Analysis and Handling #####
cat("\nStep 1.2: Missing Value Analysis and Handling\n")
cat("--------------------------------------------\n")

# Check for missing values in each column
cat("Missing values by column:\n")
missing_summary <- data.frame(
  Column = names(data),
  Missing_Count = sapply(data, function(x) sum(is.na(x))),
  Missing_Percent = round(sapply(data, function(x) sum(is.na(x)) / nrow(data) * 100), 4)
)
kable(missing_summary, caption = "Summary of Missing Values")  

# Calculate missingness proportions before handling
missing_before <- sapply(data, function(x) sum(is.na(x)) / nrow(data))
missing_before_df <- data.frame(
  Column = names(data),
  Proportion = missing_before,
  Stage = "Before Handling"
)

# Handle missing TotalCharges based on tenure
cat("\nHandling missing values in TotalCharges:\n")
cat("Rows with missing TotalCharges:\n")
na_totalcharges <- data[is.na(data$TotalCharges), c("tenure", "MonthlyCharges", "TotalCharges")]
print(na_totalcharges)

# Validate that all missing TotalCharges rows have tenure = 0
if (any(na_totalcharges$tenure != 0)) {
  cat("Warning: Some missing TotalCharges rows have tenure != 0. These will be imputed using MonthlyCharges * tenure.\n")
}

# Impute TotalCharges: If tenure = 0, set to 0; otherwise, use MonthlyCharges * tenure as an estimate
data$TotalCharges <- ifelse(is.na(data$TotalCharges) & data$tenure == 0, 0,
                            ifelse(is.na(data$TotalCharges), data$MonthlyCharges * data$tenure, data$TotalCharges))
cat("Missing TotalCharges imputed: Set to 0 where tenure = 0; otherwise, used MonthlyCharges * tenure. Remaining NAs:", sum(is.na(data$TotalCharges)), "\n")

# Calculate missingness proportions after handling
missing_after <- sapply(data, function(x) sum(is.na(x)) / nrow(data))
missing_after_df <- data.frame(
  Column = names(data),
  Proportion = missing_after,
  Stage = "After Handling"
)

# Combine before and after data for visualization
missing_comparison_df <- rbind(missing_before_df, missing_after_df)

# Visualize missing values before and after handling in a combined chart
cat("\nVisualizing missing value patterns (before and after handling):\n")

# Set a more appealing color palette with better contrast
before_color <- "#FF7676"  
after_color <- "#4A86E8"   

# Create the plot with enhanced aesthetics
ggplot(missing_comparison_df, aes(x = reorder(Column, -Proportion), y = Proportion, fill = Stage)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.9) +
  scale_y_continuous(
    labels = scales::percent_format(scale = 100), 
    limits = c(0, max(missing_comparison_df$Proportion) * 1.1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Missing Value Proportions: Before vs After Handling",
    subtitle = "Comparison of missing data across columns in the dataset",
    x = NULL,  # Remove x-axis label as column names are self-explanatory
    y = "Proportion Missing (%)",
    fill = "Data Stage",
    caption = "Note: TotalCharges initially had 0.15% missing values (11 rows)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray30"),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", margin = ggplot2::margin(t = 10)),
    axis.text.y = element_text(face = "bold"),
    axis.title.y = element_text(margin = ggplot2::margin(r = 10), face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.background = element_rect(fill = "white", color = "gray90"),
    legend.margin = ggplot2::margin(t = 5, r = 5, b = 5, l = 5),
    panel.grid.major.y = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.caption = element_text(hjust = 1, face = "italic", size = 9, color = "gray40"),
    plot.margin = ggplot2::margin(t = 10, r = 20, b = 10, l = 10)
  ) +
  scale_fill_manual(values = c("Before Handling" = before_color, "After Handling" = after_color)) +
  coord_cartesian(clip = "off")

cat("Enhanced visualization complete: improved colors, organization, and readability.\n")

# Summarize missing value handling for transparency
missing_handling_summary <- data.frame(
  Issue_Type = "Missing Values",
  Affected_Columns = "TotalCharges",
  Proportion = "0.15% (11 rows)",
  Action_Taken = "Imputed with 0 where tenure = 0; MonthlyCharges * tenure otherwise",
  Post_Action_Status = "0 missing values, 0% data loss"
)
kable(missing_handling_summary, caption = "Summary of Missing Value Handling")

##### Step 1.3: Categorical Variable Validation #####
cat("\nStep 1.3: Categorical Variable Validation\n")
cat("-----------------------------------------\n")

# Define key categorical columns (excluding customerID)
key_categorical_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
key_categorical_cols <- key_categorical_cols[key_categorical_cols != "customerID"]  

# Summarize unique values and validate
categorical_validation <- data.frame(
  Variable = character(),
  Unique_Values = character(),
  Error_Status = character(),
  Action_Taken = character(),
  stringsAsFactors = FALSE
)

# Validate specific columns
for (col in key_categorical_cols) {
  unique_vals <- unique(data[[col]])
  error_status <- "None"
  action_taken <- "Validated"
  
  # Gender: 
  if (col == "gender") {
    if (length(setdiff(unique_vals, c("Male", "Female"))) > 0) {
      error_status <- "Inconsistent values found"
      action_taken <- "Manual review required"
      cat("Warning: Inconsistent values found in gender:", unique_vals, "\n")
    }
  }
  
  # Contract: 
  if (col == "Contract") {
    valid_contracts <- c("Month-to-month", "One year", "Two year")
    if (length(setdiff(unique_vals, valid_contracts)) > 0) {
      error_status <- "Inconsistent values found"
      action_taken <- "Manual review required"
      cat("Warning: Inconsistent values found in Contract:", unique_vals, "\n")
    }
  }
  
  # PaymentMethod: Expected values are "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
  if (col == "PaymentMethod") {
    valid_payments <- c("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")
    if (length(setdiff(unique_vals, valid_payments)) > 0) {
      error_status <- "Inconsistent values found"
      action_taken <- "Manual review required"
      cat("Warning: Inconsistent values found in PaymentMethod:", unique_vals, "\n")
    }
  }
  
  # Binary Yes/No columns, accounting for dependencies
  binary_cols <- c("Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", 
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn")
  if (col %in% binary_cols) {
    # Strictly binary columns 
    strict_binary <- c("Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn")
    if (col %in% strict_binary) {
      if (length(setdiff(unique_vals, c("Yes", "No"))) > 0) {
        error_status <- "Inconsistent values found"
        action_taken <- "Manual review required"
        cat("Warning: Inconsistent values found in", col, ":", unique_vals, "\n")
      }
    }
    
    # MultipleLines: Depends on PhoneService
    if (col == "MultipleLines") {
      invalid_multiple_lines <- any(data$MultipleLines == "No phone service" & data$PhoneService != "No")
      non_no_phone_service_vals <- unique(data$MultipleLines[data$MultipleLines != "No phone service"])
      invalid_binary <- length(setdiff(non_no_phone_service_vals, c("Yes", "No"))) > 0
      if (invalid_multiple_lines || invalid_binary) {
        error_status <- "Inconsistent values found"
        action_taken <- "Manual review required"
        cat("Warning: Inconsistent values found in", col, ":", unique_vals, "\n")
      }
    }
    
    # Internet-dependent columns: Depend on InternetService
    internet_dependent_cols <- c("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
    if (col %in% internet_dependent_cols) {
      invalid_internet_service <- any(data[[col]] == "No internet service" & data$InternetService != "No")
      non_no_internet_vals <- unique(data[[col]][data[[col]] != "No internet service"])
      invalid_binary <- length(setdiff(non_no_internet_vals, c("Yes", "No"))) > 0
      if (invalid_internet_service || invalid_binary) {
        error_status <- "Inconsistent values found"
        action_taken <- "Manual review required"
        cat("Warning: Inconsistent values found in", col, ":", unique_vals, "\n")
      }
    }
  }
  
  # Add to summary table
  categorical_validation <- rbind(categorical_validation, data.frame(
    Variable = col,
    Unique_Values = paste(unique_vals, collapse = ", "),
    Error_Status = error_status,
    Action_Taken = action_taken,
    stringsAsFactors = FALSE
  ))
}

# Display validation results
kable(categorical_validation, caption = "Validation of Categorical Variables")

##### Step 1.4: Class Imbalance Check #####
cat("\nStep 1.4: Class Imbalance Check\n")
cat("--------------------------------\n")

# Check class imbalance in the target variable (Churn)
cat("Churn class distribution:\n")
churn_dist <- table(data$Churn)
churn_prop <- prop.table(churn_dist) * 100
churn_summary <- data.frame(
  Class = names(churn_dist),
  Count = as.numeric(churn_dist),
  Proportion = round(as.numeric(churn_prop), 1)
)
kable(churn_summary, caption = "Churn Class Distribution")  

# Visualize Churn imbalance with corrected geom_text
churn_plot_data <- as.data.frame(churn_dist)  
colnames(churn_plot_data) <- c("Churn", "Count")  
churn_plot_data$Proportion <- round(churn_prop, 1)  

# Calculate total for percentage labels
total_customers <- sum(churn_plot_data$Count)
churn_plot_data$Label <- paste0(format(churn_plot_data$Count, big.mark=","), 
                                " (", churn_plot_data$Proportion, "%)")

# Set custom colors with better contrast and appeal
no_color <- "#3E7DCC"   
yes_color <- "#FF6B6B"  

ggplot(churn_plot_data, aes(x = reorder(Churn, -Count), y = Count, fill = Churn)) +
  # Add gradient to bars for 3D effect
  geom_bar(stat = "identity", width = 0.7, color = "white", alpha = 0.9) +
  # Add count and percentage labels with better positioning
  geom_text(aes(label = Label), 
            position = position_stack(vjust = 0.5),
            color = "white", fontface = "bold", size = 4.5) +
  # Add count values at top of bars
  labs(
    title = "Customer Churn Distribution",
    subtitle = paste0("Total Customers: ", format(total_customers, big.mark=",")),
    x = NULL,
    y = "Number of Customers",
    caption = "Note: Approximately 3 out of 4 customers remain (No Churn)"
  ) +
  # Set y-axis to start at 0 with no padding
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)),
                     labels = scales::comma) +
  # Custom theme with better visual appeal
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray92"),
    axis.text.x = element_text(size = 13, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    axis.title.y = element_text(margin = ggplot2::margin(r = 10), face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, 
                              margin = ggplot2::margin(b = 10)),
    plot.subtitle = element_text(hjust = 0.5, color = "gray30", 
                                 margin = ggplot2::margin(b = 20)),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray40"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = ggplot2::margin(t = 20, r = 20, b = 20, l = 20)
  ) +
  # Custom color palette
  scale_fill_manual(values = c("No" = no_color, "Yes" = yes_color)) +
  # Remove legend since we have direct labels
  guides(fill = "none")

##### Step 1.5: Numerical Variable Analysis #####
cat("\nStep 1.5: Numerical Variable Analysis\n")
cat("--------------------------------------\n")

# Generate summary statistics for numerical variables
cat("Summary statistics for numerical variables:\n")
numerical_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
summary(data[, numerical_cols])  

# Check for potential outliers, extreme values, skewness, and kurtosis in numerical variables
cat("\nEnhanced outlier, extreme value, skewness, and kurtosis detection for numerical variables:\n")
outlier_summary <- data.frame(
  Column = numerical_cols,
  IQR_Outlier_Count = sapply(data[numerical_cols], function(x) {
    q <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
    iqr <- IQR(x, na.rm = TRUE)
    sum(x < q[1] - 1.5 * iqr | x > q[2] + 1.5 * iqr, na.rm = TRUE)
  }),
  Zscore_Outlier_Count = sapply(data[numerical_cols], function(x) {
    z_scores <- scale(x, center = TRUE, scale = TRUE)
    sum(abs(z_scores) > 3, na.rm = TRUE)
  }),
  Extreme_Top_1Percent_Count = sapply(data[numerical_cols], function(x) {
    upper_bound <- quantile(x, 0.99, na.rm = TRUE)
    sum(x > upper_bound, na.rm = TRUE)
  }),
  Extreme_Bottom_1Percent_Count = sapply(data[numerical_cols], function(x) {
    lower_bound <- quantile(x, 0.01, na.rm = TRUE)
    sum(x < lower_bound, na.rm = TRUE)
  }),
  Skewness = round(sapply(data[numerical_cols], function(x) skewness(x, na.rm = TRUE)), 2),
  Kurtosis = round(sapply(data[numerical_cols], function(x) kurtosis(x, na.rm = TRUE)), 2)
)
kable(outlier_summary, caption = "Outlier, Extreme Value, Skewness, and Kurtosis Detection in Numerical Variables")

# Visualize potential outliers with boxplots, adding skewness and kurtosis annotations
custom_palette <- colorRampPalette(c("#3A6B9F", "#5D9DD0", "#7BBCE8"))(length(numerical_cols))
numerical_melted <- melt(data[, numerical_cols], variable.name = "Variable", value.name = "Value")
numerical_melted$Variable <- factor(numerical_melted$Variable, levels = c("tenure", "MonthlyCharges", "TotalCharges"))
skew_kurt_labels <- outlier_summary %>%
  reframe(Variable = Column,
          Label = paste("Skewness:", round(Skewness, 2), "\nKurtosis:", round(Kurtosis, 2)),
          y_pos = Inf)
skew_kurt_labels$Variable <- factor(skew_kurt_labels$Variable, levels = c("tenure", "MonthlyCharges", "TotalCharges"))

ggplot(data = numerical_melted, aes(x = Variable, y = Value)) +
  geom_boxplot(aes(fill = Variable), width = 0.7, outlier.shape = 16,
               outlier.color = "#E74C3C", outlier.size = 1.5) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2,
               fill = "white", color = "black") +
  geom_text(data = skew_kurt_labels, aes(x = Variable, y = y_pos, label = Label),
            vjust = 1.5, size = 3.5, fontface = "bold", color = "#2C3E50") +
  scale_fill_manual(values = custom_palette) +
  labs(
    title = "Distribution of Numerical Variables (No Transformation Needed)",
    subtitle = "Boxplots with mean values (◆) and extreme values (top/bottom 1%)",
    x = NULL,
    y = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 10)),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = ggplot2::margin(b = 20)),
    plot.caption = element_text(hjust = 1, face = "italic", size = 9),
    axis.title.y = element_text(face = "bold", margin = ggplot2::margin(r = 10)),
    panel.grid.major = element_line(color = "#ECECEC"),
    panel.grid.minor = element_line(color = "#F5F5F5"),
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11),
    strip.background = element_rect(fill = "#E8F4F8", color = NA)
  ) +
  facet_wrap(~ Variable, scales = "free_y", ncol = 3) +
  theme(
    panel.border = element_rect(color = "#D5DBDB", fill = NA, size = 0.5)
  )

# Visualize distributions with histograms (no transformation applied)
cat("\nVisualizing numerical variable distributions:\n")

# Melt the data for plotting
numerical_melted <- melt(data[, numerical_cols], variable.name = "Variable", value.name = "Value")

# Melt the data for plotting
numerical_melted <- melt(data[, numerical_cols], variable.name = "Variable", value.name = "Value")

# Create histograms
ggplot(numerical_melted, aes(x = Value, fill = Variable)) +
  geom_histogram(alpha = 0.6, bins = 30) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(
    title = "Histograms of Numerical Variables",
    subtitle = "Distributions of tenure, MonthlyCharges, and TotalCharges",
    x = "Value", 
    y = "Count",
    caption = "Note: No transformation applied as skewness is within stable range."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 10)),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = ggplot2::margin(b = 20)),
    plot.caption = element_text(hjust = 1, face = "italic", size = 9),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11),
    strip.background = element_rect(fill = "#E8F4F8", color = NA),
    panel.border = element_rect(color = "#D5DBDB", fill = NA, size = 0.5),
    legend.position = "none"  # Remove legend since facet labels are sufficient
  ) +
  scale_fill_manual(values = colorRampPalette(c("#3A6B9F", "#5D9DD0", "#7BBCE8"))(length(numerical_cols)))

# Summary of outlier management
cat("\nOutlier Management Summary:\n")
cat("No IQR or z-score outliers detected. Skewness within stable range (",
    paste(outlier_summary$Skewness, collapse = ", "),
    "). No transformation applied.\n")

# Conclusion
cat("\nStep 1 complete: Initial inspection, error correction, and numerical analysis performed. No categorical errors, missing values, or significant outliers remain. Class imbalance identified and will be handled in a later step.\n")

############# Step 2: Categorical Variable Encoding ################
cat("\nStep 2: Categorical Variable Encoding\n")
cat("-------------------------------------\n")

# Identify categorical columns (excluding customerID)
categorical_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
categorical_cols <- categorical_cols[categorical_cols != "customerID"]

# Encode binary Yes/No columns as 0/1
binary_cols <- c("Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", 
                 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                 "PaperlessBilling", "Churn")
for (col in binary_cols) {
  # Convert "No phone service" or "No internet service" to "No" for consistency
  data[[col]] <- ifelse(data[[col]] %in% c("No phone service", "No internet service"), "No", data[[col]])
  # Encode Yes/No as 1/0
  data[[col]] <- ifelse(data[[col]] == "Yes", 1, 0)
}

# Re-identify categorical columns after binary encoding (since binary_cols are now numeric)
categorical_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
categorical_cols <- categorical_cols[categorical_cols != "customerID"]

# Apply one-hot encoding for multi-category variables (e.g., gender, Contract, PaymentMethod, InternetService)
multi_category_cols <- setdiff(categorical_cols, binary_cols)
if (length(multi_category_cols) > 0) {
  # Ensure all multi-category columns are factors
  for (col in multi_category_cols) {
    if (!is.factor(data[[col]])) {
      data[[col]] <- as.factor(data[[col]])
    }
  }
  
  # Create dummy variables using model.matrix
  dummy_vars <- model.matrix(~ . - 1, data = data[multi_category_cols], 
                             contrasts.arg = lapply(data[multi_category_cols], contrasts, contrasts = FALSE))
  dummy_df <- as.data.frame(dummy_vars)
  
  # Remove original multi-category columns and add dummy variables
  data <- data[, !names(data) %in% multi_category_cols]
  data <- cbind(data, dummy_df)
}

# Summarize encoding
cat("Categorical variables encoded:\n")
cat("- Binary variables (", paste(binary_cols, collapse = ", "), "): Encoded as 0/1 (Yes = 1, No = 0)\n")
cat("- Multi-category variables (", paste(multi_category_cols, collapse = ", "), "): One-hot encoded\n")
cat("Updated dataset dimensions after encoding:", dim(data), "\n")

############# Step 3: Standardizing Numerical Variables ################
cat("\nStep 3: Standardizing Numerical Variables\n")
cat("-----------------------------------------\n")

# Identify numerical columns
numerical_cols <- c("tenure", "MonthlyCharges", "TotalCharges")

# Save original numerical variables before scaling
data$tenure_original <- data$tenure
data$MonthlyCharges_original <- data$MonthlyCharges
data$TotalCharges_original <- data$TotalCharges

# Standardize numerical variables
data[numerical_cols] <- scale(data[numerical_cols])

# Verify standardization (mean ~0, sd ~1)
cat("Summary of standardized numerical variables:\n")
summary_stats <- do.call(cbind, lapply(data[numerical_cols], function(x) c(Mean = round(mean(x), 4), SD = round(sd(x), 4))))
kable(summary_stats, caption = "Mean and SD of Standardized Numerical Variables")

############# Step 4: Feature Engineering ################
cat("\nStep 4: Feature Engineering\n")
cat("---------------------------\n")

# Example: Create a new feature for average monthly charges
data$AvgMonthlyCharges <- ifelse(data$tenure == 0, data$MonthlyCharges, data$TotalCharges / data$tenure)

# Feature Engineering: Fixed tenure bins
data$TenureGroup <- cut(
  data$tenure_original, 
  breaks = c(-Inf, 12, 24, Inf), 
  labels = c("Short-term", "Medium-term", "Long-term")
)

# Apply one-hot encoding to the new categorical feature (TenureGroup)
dummy_vars <- model.matrix(~ TenureGroup - 1, data = data, 
                           contrasts.arg = list(TenureGroup = contrasts(data$TenureGroup, contrasts = FALSE)))
dummy_df <- as.data.frame(dummy_vars)
colnames(dummy_df) <- gsub("TenureGroup", "Tenure_", colnames(dummy_df))  
data <- data[, !names(data) %in% "TenureGroup"]  
data <- cbind(data, dummy_df)  

# Summarize new features
cat("Summary of new features:\n")
kable(summary(data[, c("AvgMonthlyCharges", grep("Tenure_", names(data), value = TRUE))]), 
      caption = "New Features Summary")

# Update numerical_cols for standardization in case new numerical features are added
numerical_cols <- c(numerical_cols, "AvgMonthlyCharges")

# Standardize the new numerical feature
data$AvgMonthlyCharges <- scale(data$AvgMonthlyCharges)

# Verify standardization of the new feature
cat("\nSummary of standardized new numerical feature:\n")
kable(data.frame(Mean = round(mean(data$AvgMonthlyCharges), 4), SD = round(sd(data$AvgMonthlyCharges), 4)), 
      caption = "Mean and SD of Standardized AvgMonthlyCharges")

############# Step 5: Final Quality Check ################
cat("\nStep 5: Final Quality Check\n")
cat("---------------------------\n")

# Check for missing values
cat("Missing values after preprocessing:\n")
missing_final <- sapply(data, function(x) sum(is.na(x)))
kable(data.frame(Column = names(missing_final), Missing = missing_final), caption = "Final Missing Value Check")

# Confirm all columns are numeric (except customerID if still present)
cat("\nVariable types after preprocessing:\n")
var_types_final <- sapply(data, class)
kable(data.frame(Column = names(var_types_final), Type = as.character(var_types_final)), caption = "Final Variable Types")

# Final dimensions and class distribution (pre-SMOTE)
cat("\nFinal dataset dimensions:", dim(data), "\n")
cat("Final Churn distribution (before SMOTE):\n")
kable(table(data$Churn), caption = "Final Churn Distribution")

############# Step 6: Export Processed Data ################
cat("\nStep 6: Export Processed Data\n")
cat("-----------------------------\n")

# Define the output file path
output_file <- "data/telco_processed.csv"

# Export the processed dataset to a CSV file
write.csv(data, file = output_file, row.names = FALSE)

################################################################################
##################### Task 3: Data visualization ###############################
################################################################################

############# Figure 1:Churn Rate by Tenure Group & Contract Type ###############
# Reconstruct the Contract column from one-hot encoded columns
data <- data %>%
  mutate(
    Contract = case_when(
      `ContractMonth-to-month` == 1 ~ "Month-to-Month",
      `ContractOne year` == 1 ~ "One Year",
      `ContractTwo year` == 1 ~ "Two Year",
      TRUE ~ NA_character_  
    ),
    # Reconstruct TenureGroup from one-hot encoded columns
    TenureGroup = case_when(
      `Tenure_Short-term` == 1 ~ "Short-term",
      `Tenure_Medium-term` == 1 ~ "Medium-term",
      `Tenure_Long-term` == 1 ~ "Long-term",
      TRUE ~ NA_character_  
    )
  )

# Calculate churn rate by TenureGroup and Contract
churn_summary <- data %>%
  group_by(Contract, TenureGroup) %>%
  summarise(
    ChurnRate = mean(Churn, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  ) %>%
  mutate(
    Label = paste0(round(ChurnRate * 100, 1), "%\n(n=", Count, ")")
  ) %>%
  filter(!is.na(Contract) & !is.na(TenureGroup))  

# Define a modern color palette
churn_color <- "#E63946"  
no_churn_color <- "#457B9D"  

# p1: Churn Rate by Tenure Group & Contract Type
p1 <- ggplot(churn_summary, aes(x = TenureGroup, y = ChurnRate, fill = TenureGroup)) +
  geom_bar(stat = "identity", alpha = 0.9, width = 0.45) +
  geom_text(aes(label = Label), vjust = -0.5, size = 3.5, fontface = "bold", color = "#2D3436") +
  facet_wrap(~ Contract, ncol = 3, scales = "free_x") +
  scale_y_continuous(labels = percent_format(scale = 100), limits = c(0, max(churn_summary$ChurnRate, na.rm = TRUE) * 1.2)) +
  scale_fill_manual(values = c("Short-term" = "#F94144", "Medium-term" = "#F3722C", "Long-term" = "#F8961E")) +
  scale_x_discrete(labels = c("Short-term" = "Short-term\n(≤ 12 months)", 
                              "Medium-term" = "Medium-term\n(12–24 months)", 
                              "Long-term" = "Long-term\n(> 24 months)")) +
  labs(
    title = "Churn Rate by Tenure Group and Contract Type",
    subtitle = "Impact of Tenure Length Across Contract Durations",
    x = "Tenure Group",
    y = "Churn Rate (%)",
    caption = "Note: Churn rates calculated using engineered TenureGroup feature. Sample sizes shown in labels.",
    fill = "Tenure Group"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 12),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray40"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold", size = 10),
    axis.text.y = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11),
    strip.background = element_rect(fill = "#ECECEC", color = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    plot.margin = ggplot2::margin(t = 10, r = 20, b = 10, l = 10)
  )

print(p1)

############# Figure 2: Monthly Charges Distribution by Churn Status ###############
# Prepare data for visualization
data$ChurnStatus <- factor(data$Churn, levels = c(0, 1), labels = c("No Churn", "Churn"))

# Calculate mean monthly charges for each group and the difference
mean_charges <- data %>%
  group_by(ChurnStatus) %>%
  summarise(
    MeanCharges = mean(MonthlyCharges_original, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  )

# Calculate the average difference
avg_diff <- mean_charges$MeanCharges[mean_charges$ChurnStatus == "Churn"] - 
  mean_charges$MeanCharges[mean_charges$ChurnStatus == "No Churn"]
avg_diff_text <- paste("Average Difference: $", round(abs(avg_diff), 2), sep = "")

# Prepare labels for sample sizes
mean_charges <- mean_charges %>%
  mutate(Label = paste0("n=", Count))

# Define a modern color palette
churn_color <- "#E63946"  
no_churn_color <- "#457B9D"  

# p2: Monthly Charges by Churn Status (Violin + Boxplot)
p2 <- ggplot(data, aes(x = ChurnStatus, y = MonthlyCharges_original, fill = ChurnStatus)) +
  geom_violin(alpha = 0.7, width = 0.45, trim = FALSE) +
  geom_boxplot(width = 0.15, fill = "white", color = "#2D3436", 
               outlier.shape = 16, outlier.size = 1.5, outlier.color = "#2D3436") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2, 
               fill = "#F7DC6F", color = "#2D3436") +
  geom_text(data = mean_charges, aes(x = ChurnStatus, y = max(data$MonthlyCharges_original, na.rm = TRUE) * 1.1, 
                                     label = Label), 
            vjust = -0.5, size = 4, fontface = "bold", color = "#2D3436") +
  scale_y_continuous(labels = dollar_format(prefix = "$"), 
                     breaks = seq(0, max(data$MonthlyCharges_original, na.rm = TRUE), by = 20),
                     limits = c(0, max(data$MonthlyCharges_original, na.rm = TRUE) * 1.2)) +
  scale_fill_manual(values = c("No Churn" = no_churn_color, "Churn" = churn_color)) +
  labs(
    title = "Monthly Charges Distribution by Churn Status",
    subtitle = "Comparing Spending Patterns of Churners vs. Loyal Customers",
    x = "Churn Status",
    y = "Monthly Charges ($)",
    caption = paste(avg_diff_text, 
                    "\nNote: Violin plots show density; boxplots highlight medians and outliers. ♦ indicates mean value."),
    fill = "Churn Status"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5, family = "Arial"),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 14, family = "Arial"),
    plot.caption = element_text(hjust = 1, face = "italic", size = 11, color = "gray40", family = "Arial"),
    axis.title = element_text(face = "bold", size = 14, family = "Arial"),
    axis.text = element_text(face = "bold", size = 12, color = "#2D3436", family = "Arial"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray90", linetype = "dashed"),
    panel.background = element_rect(fill = "#F9FAFB", color = NA),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 12, family = "Arial"),
    legend.text = element_text(size = 11, family = "Arial"),
    legend.background = element_rect(fill = "white", color = "gray90"),
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
    plot.margin = ggplot2::margin(t = 15, r = 25, b = 15, l = 25)
  )

print(p2)

############# Figure 3: Feature Correlation and Variable Importance Heatmap ###############
# Rename columns to replace special characters (hyphens, spaces, parentheses) with underscores
names(data) <- gsub("[- ()]", "_", names(data))

# Correlation matrix for numerical variables
numerical_cols1 <- c("tenure_original", "MonthlyCharges_original", "TotalCharges_original", "AvgMonthlyCharges")
cor_matrix1 <- cor(data[, numerical_cols], use = "complete.obs")
cor_melted1 <- melt(cor_matrix1)
cor_melted1$value <- round(cor_melted1$value, 2)

# Simulate variable importance using random forest
# Update column names in the grep pattern to match the renamed columns
set.seed(123)
rf_model1 <- randomForest(Churn ~ ., 
                         data = data[, c("Churn", numerical_cols, 
                                         grep("Contract|PaymentMethod|Tenure_", names(data), value = TRUE))], 
                         importance = TRUE, ntree = 100)
var_imp1 <- as.data.frame(importance(rf_model1, type = 1))
var_imp1$Variable <- rownames(var_imp1)
colnames(var_imp1) <- c("Importance", "Variable")
var_imp1 <- var_imp1 %>% arrange(desc(Importance)) %>% head(10)  
var_imp1$Variable <- factor(var_imp1$Variable, levels = var_imp1$Variable)

# p3: Correlation heatmap (viridis palette)
p3 <- ggplot(cor_melted1, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "white", size = 3.5, fontface = "bold") +
  scale_fill_viridis_c(option = "viridis", direction = -1, limits = c(-1, 1)) +
  labs(
    title = "Correlation Matrix of Numerical Features",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    panel.grid = element_blank(),
    legend.title = element_text(face = "bold"),
    legend.position = "right"
  )

# p4: Variable importance plot (viridis tone)
p4 <- ggplot(var_imp1, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#21908C", alpha = 0.9) +  
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Top 10 Variable Importance for Churn Prediction",
    x = NULL, y = "Mean Decrease in Accuracy",
    caption = "Note: Variable importance derived from Random Forest model."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray40"),
    axis.text = element_text(face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

# p3+p4: Correlation and Variable Importance
combined_plot <- p3 + p4 + plot_layout(ncol = 2, widths = c(1, 1)) +
  plot_annotation(
    title = "Feature Correlation and Variable Importance",
    subtitle = "Linking EDA with Predictive Modeling Insights",
    caption = "Note: Correlation matrix uses original numerical features; importance includes engineered features.",
    theme = theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 12)
    )
  )

print(combined_plot)

############# Figure 4: Churn Rate by Payment Method ###############
# Clean column names (optional, but recommended for easier handling)
# Replace spaces, hyphens, and parentheses with underscores
names(data) <- gsub("[- ()]", "_", names(data))

# Verify the cleaned column names
colnames(data)

# Mutate with the corrected PaymentMethod columns
data <- data %>%
  mutate(
    PaymentMethod = case_when(
      PaymentMethodBank_transfer__automatic_ == 1 ~ "Bank Transfer",
      PaymentMethodCredit_card__automatic_ == 1 ~ "Credit Card",
      PaymentMethodElectronic_check == 1 ~ "Electronic Check",
      PaymentMethodMailed_check == 1 ~ "Mailed Check",
      TRUE ~ NA_character_  
    )
  )

# Calculate churn rate by PaymentMethod
churn_summary <- data %>%
  group_by(PaymentMethod) %>%
  summarise(
    ChurnRate = mean(Churn, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  ) %>%
  mutate(
    Label = paste0(round(ChurnRate * 100, 1), "%"),
    PaymentMethod = factor(PaymentMethod, levels = PaymentMethod[order(ChurnRate)])
  ) %>%
  filter(!is.na(PaymentMethod))  

# Define a refined color palette with stronger visual impact
color_gradient <- colorRampPalette(c("#FEE2E2", "#EF4444", "#B91C1C", "#7F1D1D"))(8)

# Figure 4: Churn Rate by Payment Method (Lollipop Chart)
p5 <- ggplot(churn_summary, aes(x = reorder(PaymentMethod, ChurnRate), y = ChurnRate, color = ChurnRate)) +
  # Lollipop segments
  geom_segment(aes(xend = PaymentMethod, yend = 0), size = 1, linetype = "solid") +
  # Lollipop points
  geom_point(size = 4, shape = 21, fill = "white", stroke = 1.5) +
  # Churn rate labels
  geom_text(aes(label = Label), hjust = -0.3, size = 4, fontface = "bold", color = "#2D3436") +
  # Sample size labels below each point
  geom_text(aes(y = -0.02, label = paste0("(n=", Count, ")")), size = 3.5, color = "gray40", vjust = 1.2) +
  # Flip coordinates for horizontal lollipop chart
  coord_flip() +
  # Scale and format y-axis as percentage
  scale_y_continuous(
    labels = scales::percent_format(scale = 100),
    limits = c(-0.05, max(churn_summary$ChurnRate, na.rm = TRUE) * 1.2),
    breaks = seq(0, max(churn_summary$ChurnRate, na.rm = TRUE), by = 0.1)
  ) +
  # Apply color gradient based on churn rate
  scale_color_gradientn(colors = color_gradient, guide = "none") +
  # Customize labels and captions
  labs(
    title = "Churn Rate by Payment Method",
    subtitle = "Ranking Payment Methods by Customer Churn Risk",
    x = NULL,
    y = "Churn Rate (%)",
    caption = "Note: Churn rates calculated using one-hot encoded PaymentMethod features.\nElectronic Check shows notably higher churn risk, suggesting targeted retention strategies."
  ) +
  # Apply a clean, modern theme
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = ggplot2::margin(t = 10, r = 0, b = 10, l = 0)),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 12, margin = ggplot2::margin(t = 15, r = 0, b = 10, l = 0)),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray40", margin = ggplot2::margin(t = 10, r = 0, b = 0, l = 0)),
    axis.title.x = element_text(face = "bold", size = 12, margin = ggplot2::margin(t = 10, r = 0, b = 0, l = 0)),
    axis.text.x = element_text(face = "bold", size = 10),
    axis.text.y = element_text(face = "bold", size = 11, color = "#2D3436"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "gray90", linetype = "dashed"),
    panel.background = element_rect(fill = "#F9FAFB", color = NA),
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
    plot.margin = ggplot2::margin(t = 15, r = 25, b = 15, l = 25)
  )

# Print the plot
print(p5)

################################################################################
##################### Task 4: Predictive Modeling ##############################
################################################################################

# Set seed for reproducibility
set.seed(123)

# Display structure of the dataset
str(data)

# Duplicate dataset
data_predictive <- data

# Remove unwanted columns
data_predictive <- data_predictive %>%
  select(-customerID, -tenure_original, -MonthlyCharges_original, -TotalCharges_original, 
         -Contract, -TenureGroup, -ChurnStatus, -PaymentMethod)

# Ensure Churn is a factor for classification
data_predictive$Churn <- factor(data_predictive$Churn, levels = c(0, 1), labels = c("No", "Yes"))

############# Step 1: Train/Test Split ################
cat("\nStep 1: Train/Test Split\n")
cat("------------------------\n")

# Split data into 70% training and 30% testing
trainIndex <- createDataPartition(data_predictive$Churn, p = 0.7, list = FALSE)
train_data <- data_predictive[trainIndex, ]
test_data <- data_predictive[-trainIndex, ]

# Verify split proportions
cat("Training set dimensions:", dim(train_data), "\n")
cat("Testing set dimensions:", dim(test_data), "\n")
cat("Churn distribution in training set:\n")
kable(table(train_data$Churn), caption = "Churn Distribution in Training Set")
cat("Churn distribution in testing set:\n")
kable(table(test_data$Churn), caption = "Churn Distribution in Testing Set")

############# Step 2: Address Class Imbalance (SMOTE) ################
cat("\nStep 2: Address Class Imbalance with SMOTE\n")
cat("-----------------------------------------\n")

# Apply SMOTE to the training set using smotefamily
minority_count <- sum(train_data$Churn == "Yes")
majority_count <- sum(train_data$Churn == "No")

# Target: Achieve 60:40 ratio (Yes:No)
target_minority_count <- majority_count * (60 / 40)  # For 60:40 balance
synthetic_samples_needed <- target_minority_count - minority_count
dup_size_needed <- ceiling(synthetic_samples_needed / minority_count)

# Adjust dup_size to ensure we don't overshoot (manually set to 2 based on calculation)
dup_size_needed <- 2

# Apply SMOTE with corrected dup_size
smote_result <- SMOTE(
  X = train_data[, !names(train_data) %in% "Churn"], 
  target = train_data$Churn,                        
  K = 5,                                           
  dup_size = dup_size_needed                        
)

# Extract the SMOTE-balanced dataset
smote_train <- smote_result$data
colnames(smote_train)[ncol(smote_train)] <- "Churn" 
smote_train$Churn <- factor(smote_train$Churn, levels = c("No", "Yes"))

# Verify new class distribution after SMOTE
cat("Churn distribution after SMOTE:\n")
kable(table(smote_train$Churn), caption = "Churn Distribution After SMOTE")

############# Step 3: Train Base Models ################
cat("\nStep 3: Train Base Models\n")
cat("-------------------------\n")

# Define predictors (all columns except Churn)
predictors <- setdiff(names(smote_train), "Churn")

# Logistic Regression 
cat("Training Logistic Regression model...\n")
logit_model <- glm(Churn ~ ., data = smote_train, family = binomial(link = "logit"))

# Random Forest 
cat("Training Random Forest model with hyperparameter tuning...\n")
rf_control <- trainControl(
  method = "cv",
  number = 5,
  search = "grid",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
rf_model <- train(
  Churn ~ .,
  data = smote_train,
  method = "rf",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 100
)

# Add XGBoost 
cat("Training XGBoost model...\n")
xgb_control <- trainControl(
  method = "cv",
  number = 3,  
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
xgb_grid <- expand.grid(
  nrounds = 50,  # Reduce boosting rounds
  max_depth = c(3, 6),
  eta = 0.3,  # Use a single, higher learning rate for faster convergence
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)
xgb_model <- train(
  Churn ~ .,
  data = smote_train,
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE  # Suppress verbose output for cleaner logs
)

# Display best parameters for Random Forest and XGBoost
cat("Best Random Forest parameters:\n")
kable(rf_model$bestTune, caption = "Optimal Random Forest Hyperparameters")
cat("Best XGBoost parameters:\n")
kable(xgb_model$bestTune, caption = "Optimal XGBoost Hyperparameters")

############# Step 4: Predict Probabilities ################
cat("\nStep 4: Predict Probabilities\n")
cat("----------------------------\n")

# Logistic Regression predictions
logit_test_probs <- predict(logit_model, newdata = test_data, type = "response")

# Random Forest predictions
rf_test_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "Yes"]

# XGBoost predictions
xgb_test_probs <- predict(xgb_model, newdata = test_data, type = "prob")[, "Yes"]

############# Step 5: Model Evaluation ################
cat("\nStep 5: Model Evaluation\n")
cat("-----------------------\n")

# Evaluate models on test set
cat("Model performance on test set:\n")

# Logistic Regression
logit_test_roc <- roc(test_data$Churn, logit_test_probs, levels = c("No", "Yes"), direction = "<")
logit_auc <- auc(logit_test_roc)
logit_cm <- confusionMatrix(factor(ifelse(logit_test_probs > 0.5, "Yes", "No"), levels = c("No", "Yes")), test_data$Churn)
logit_f1 <- logit_cm$byClass["F1"]
logit_sensitivity <- logit_cm$byClass["Sensitivity"]
logit_specificity <- logit_cm$byClass["Specificity"]

# Random Forest
rf_test_roc <- roc(test_data$Churn, rf_test_probs, levels = c("No", "Yes"), direction = "<")
rf_auc <- auc(rf_test_roc)
rf_cm <- confusionMatrix(predict(rf_model, newdata = test_data), test_data$Churn)
rf_f1 <- rf_cm$byClass["F1"]
rf_sensitivity <- rf_cm$byClass["Sensitivity"]
rf_specificity <- rf_cm$byClass["Specificity"]

# XGBoost
xgb_test_roc <- roc(test_data$Churn, xgb_test_probs, levels = c("No", "Yes"), direction = "<")
xgb_auc <- auc(xgb_test_roc)
xgb_cm <- confusionMatrix(predict(xgb_model, newdata = test_data), test_data$Churn)
xgb_f1 <- xgb_cm$byClass["F1"]
xgb_sensitivity <- xgb_cm$byClass["Sensitivity"]
xgb_specificity <- xgb_cm$byClass["Specificity"]

# Summarize performance
performance_summary <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC = round(c(logit_auc, rf_auc, xgb_auc), 3),
  Accuracy = round(c(logit_cm$overall["Accuracy"], rf_cm$overall["Accuracy"], xgb_cm$overall["Accuracy"]), 3),
  Sensitivity = round(c(logit_sensitivity, rf_sensitivity, xgb_sensitivity), 3),
  Specificity = round(c(logit_specificity, rf_specificity, xgb_specificity), 3),
  F1_Score = round(c(logit_f1, rf_f1, xgb_f1), 3)
)
kable(performance_summary, caption = "Model Performance Summary on Test Set")

# Define α, β, γ 
alpha <- 0.4  
beta <- 0.3   
gamma <- 0.2  

# Apply custom scoring formula: α*AUC + β*F1_Score + γ*Sensitivity + (1 - α - β - γ)*Specificity
performance_summary$Custom_Score <- alpha * performance_summary$AUC + beta * performance_summary$F1_Score +
  gamma * performance_summary$Sensitivity + (1 - alpha - beta - gamma) * performance_summary$Specificity

# Select the best model based on Custom Score
best_model <- performance_summary[which.max(performance_summary$Custom_Score), "Model"]
cat("Best model based on Custom Score:", best_model, "\n")

# Assign the best model object for feature importance
if (best_model == "Logistic Regression") {
  best_model_obj <- logit_model
} else if (best_model == "Random Forest") {
  best_model_obj <- rf_model
} else {
  best_model_obj <- xgb_model
}

# Confusion matrices
# Convert confusion matrices to data frames for plotting
logit_cm_df <- as.data.frame(as.table(logit_cm$table)) |> 
  setNames(c("Prediction", "Reference", "Freq"))
rf_cm_df <- as.data.frame(as.table(rf_cm$table)) |> 
  setNames(c("Prediction", "Reference", "Freq"))
xgb_cm_df <- as.data.frame(as.table(xgb_cm$table)) |> 
  setNames(c("Prediction", "Reference", "Freq"))

# Calculate accuracy for each model
logit_acc <- round(sum(diag(logit_cm$table)) / sum(logit_cm$table) * 100, 1)
rf_acc <- round(sum(diag(rf_cm$table)) / sum(rf_cm$table) * 100, 1)
xgb_acc <- round(sum(diag(xgb_cm$table)) / sum(xgb_cm$table) * 100, 1)

# Text color contrast helper
get_text_color <- function(fill_value, min_value, max_value, threshold = 0.6) {
  normalized <- (fill_value - min_value) / (max_value - min_value)
  ifelse(normalized > threshold, "white", "black")
}

# Shared color scaling
all_freqs <- c(logit_cm_df$Freq, rf_cm_df$Freq, xgb_cm_df$Freq)
min_freq <- min(all_freqs)
max_freq <- max(all_freqs)

# Elegant theme (FIXED margin calls)
elegant_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = ggplot2::margin(b = 15)),
    plot.subtitle = element_text(hjust = 0.5, size = 13, margin = ggplot2::margin(b = 10)),
    axis.title = element_text(face = "bold", size = 13, margin = ggplot2::margin(t = 10, b = 10)),
    axis.text = element_text(size = 12, color = "black"),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    plot.margin = ggplot2::margin(t = 15, r = 15, b = 15, l = 15)
  )

# Plot for Logistic Regression
plot_logit <- ggplot(logit_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq, color = get_text_color(Freq, min_freq, max_freq)), size = 5, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#E1F5FE", "#81D4FA", "#29B6F6", "#0288D1", "#01579B"),
    name = "Count", limits = c(min_freq, max_freq),
    guide = guide_colorbar(direction = "horizontal", barwidth = 10, barheight = 0.8, 
                           title.position = "top", title.hjust = 0.5)
  ) +
  scale_color_manual(values = c("black", "white"), guide = "none") +
  labs(
    title = "Logistic Regression",
    subtitle = paste0("Accuracy: ", logit_acc, "%"),
    x = "Actual Class", y = "Predicted Class"
  ) +
  elegant_theme +
  theme(legend.position = "bottom")

# Plot for Random Forest
plot_rf <- ggplot(rf_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq, color = get_text_color(Freq, min_freq, max_freq)), size = 5, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#FFF8E1", "#FFECB3", "#FFD54F", "#FFC107", "#FF8F00"),
    name = "Count", limits = c(min_freq, max_freq),
    guide = guide_colorbar(direction = "horizontal", barwidth = 10, barheight = 0.8, 
                           title.position = "top", title.hjust = 0.5)
  ) +
  scale_color_manual(values = c("black", "white"), guide = "none") +
  labs(
    title = "Random Forest",
    subtitle = paste0("Accuracy: ", rf_acc, "%"),
    x = "Actual Class", y = "Predicted Class"
  ) +
  elegant_theme +
  theme(legend.position = "bottom")

# Plot for XGBoost
plot_xgb <- ggplot(xgb_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq, color = get_text_color(Freq, min_freq, max_freq)), size = 5, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#F1F8E9", "#C5E1A5", "#9CCC65", "#7CB342", "#33691E"),
    name = "Count", limits = c(min_freq, max_freq),
    guide = guide_colorbar(direction = "horizontal", barwidth = 10, barheight = 0.8, 
                           title.position = "top", title.hjust = 0.5)
  ) +
  scale_color_manual(values = c("black", "white"), guide = "none") +
  labs(
    title = "XGBoost",
    subtitle = paste0("Accuracy: ", xgb_acc, "%"),
    x = "Actual Class", y = "Predicted Class"
  ) +
  elegant_theme +
  theme(legend.position = "bottom")

# Footer text
caption_text <- "Diagonal elements represent correct predictions. Higher values on the diagonal indicate better model performance."

# Combine all plots
combined_plot2 <- (plot_logit | plot_rf | plot_xgb) +
  plot_layout(guides = "keep") +
  plot_annotation(
    title = "Comparative Model Performance Analysis",
    subtitle = "Confusion Matrices with Distinct Color Schemes",
    caption = caption_text,
    theme = theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 10)),
      plot.subtitle = element_text(size = 16, hjust = 0.5, margin = ggplot2::margin(b = 20)),
      plot.caption = element_text(size = 12, hjust = 0, margin = ggplot2::margin(t = 15), color = "grey30", lineheight = 1.2),
      plot.background = element_rect(fill = "#FAFAFA", color = NA),
      panel.background = element_rect(fill = "#FAFAFA", color = NA)
    )
  )

# Show plot
print(combined_plot2)



############# Step 6: Visualization ################
cat("\nStep 6: Visualization\n")
cat("--------------------\n")

# Visualization 1: Comparative ROC Curve for All Models
cat("\nCreating Comparative ROC Curve for All Models...\n")

# Prepare ROC data
roc_data <- data.frame(
  Model = c(rep("Logistic Regression", length(logit_test_roc$sensitivities)),
            rep("Random Forest", length(rf_test_roc$sensitivities)),
            rep("XGBoost", length(xgb_test_roc$sensitivities))),
  Sensitivity = c(logit_test_roc$sensitivities, rf_test_roc$sensitivities, xgb_test_roc$sensitivities),
  Specificity = c(1 - logit_test_roc$specificities, 1 - rf_test_roc$specificities, 1 - xgb_test_roc$specificities)
)

# Add Highlight
roc_data$Highlight <- ifelse(roc_data$Model == best_model, "Best", "Other")

# New pastel colors
model_colors <- c(
  "Logistic Regression" = "#1a2880", 
  "Random Forest" = "#862b1d", 
  "XGBoost" = "#f8c25b"
)

# Plot
p_roc <- ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Model, group = Model)) +
  geom_line(aes(size = Highlight, alpha = Highlight)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray70") +
  
  #Color per Model
  scale_color_manual(values = model_colors, name = "Model") +
  
  #Size and alpha per Highlight
  scale_size_manual(values = c("Best" = 1.5, "Other" = 1)) +
  scale_alpha_manual(values = c("Best" = 0.8, "Other" = 0.6)) +
  
  labs(
    title = "ROC Curves for Churn Prediction Models",
    subtitle = "Comparing Logistic Regression, Random Forest, and XGBoost",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    caption = paste("AUC: Logistic Regression =", round(logit_auc, 3), 
                    "| Random Forest =", round(rf_auc, 3), 
                    "| XGBoost =", round(xgb_auc, 3))
  ) +
  theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 13),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray50"),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    plot.margin = ggplot2::margin(t = 15, r = 25, b = 15, l = 25),
    
    #Legend with larger color swatches
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 12),
    legend.background = element_rect(fill = "gray98", color = "gray90"),
    legend.key = element_rect(fill = "gray98", color = NA, size = 1.5),  
    legend.key.height = unit(1.5, "lines"),  
    legend.key.width = unit(2, "lines"),    
    legend.box = "horizontal"
  )

print(p_roc)
ggsave("outputs/roc_curves.png", p_roc, width = 10, height = 7, dpi = 150, bg = "white")

# Visualization 2: Feature Importance Plot for the Best Model (if applicable)
if (!is.null(best_model_obj)) {
  cat("\nCreating Feature Importance Plot for the Best Model...\n")
  
  # Extract feature importance
  if (best_model == "Random Forest") {
    var_imp <- as.data.frame(importance(best_model_obj$finalModel))
    var_imp$Variable <- rownames(var_imp)
    var_imp$Importance <- var_imp$MeanDecreaseGini
  } else if (best_model == "XGBoost") {
    var_imp <- varImp(best_model_obj, scale = FALSE)$importance
    var_imp$Variable <- rownames(var_imp)
    var_imp$Importance <- var_imp$Overall
  }
  
  # Prepare data
  var_imp <- var_imp %>%
    mutate(Variable = gsub("_", " ", Variable)) %>%
    arrange(desc(Importance)) %>%
    head(10) %>%
    mutate(Variable = factor(Variable, levels = rev(Variable)),
           IsEngineered = Variable %in% c("tenure"))
  
  # Plot
  p_importance <- ggplot(var_imp, aes(x = Variable, y = Importance, fill = Importance)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    coord_flip() +
    scale_fill_gradient(low = "#F4A261", high = "#264653") +
    geom_text(aes(label = sprintf("%.3f", Importance)), hjust = -0.1, size = 4.2, color = "gray20") +
    labs(
      title = paste("Top 10 Important Features for", best_model),
      subtitle = "Feature Importance for Churn Prediction",
      x = NULL,
      y = ifelse(best_model == "Random Forest", "Mean Decrease in Gini", "Importance"),
      caption = "Note: Engineered feature (tenure) is among the selected top features."
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 13),
      plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray50"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "gray90"),
      plot.margin = ggplot2::margin(t = 15, r = 30, b = 15, l = 30)
    ) +
    ylim(0, max(var_imp$Importance) * 1.15)
  
  print(p_importance)
  ggsave("outputs/feature_importance.png", p_importance, width = 10, height = 7, dpi = 150, bg = "white")
  
} else {
  cat("Feature importance plot not available for Logistic Regression.\n")
}
# Step 7: Feature Set Comparison Using AUC
# Objective: Compare model performance (focusing on AUC) using different feature sets 
# (full features vs. top 10 features) for Logistic Regression, Random Forest, and XGBoost 
# to explore iterative improvements.

cat("\nStep 7: Feature Set Comparison Using AUC\n")
cat("---------------------------------------\n")

# 1. Define feature sets
# Full feature set: All predictors except the target variable (Churn)
full_features <- setdiff(names(smote_train), "Churn")

# Top feature set: Select top 10 features based on importance from the Random Forest model
var_imp <- as.data.frame(importance(rf_model$finalModel))
var_imp$Variable <- rownames(var_imp)
var_imp <- var_imp[order(-var_imp$MeanDecreaseGini), ]  # Sort by importance
top_features <- var_imp$Variable[1:10]  # Select top 10 features
cat("Top 10 features selected for comparison:\n", paste(top_features, collapse = ", "), "\n")

# 2. Define trainControl for consistent evaluation
rf_control <- trainControl(
  method = "cv",
  number = 5,
  search = "grid",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = FALSE
)

xgb_control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = FALSE
)

# Define tuning grids (reusing your original settings)
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
xgb_grid <- expand.grid(
  nrounds = 50,
  max_depth = c(3, 6),
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# 3. Train models on different feature sets
# Logistic Regression
cat("Training Logistic Regression on full and top 10 feature sets...\n")
logit_full <- glm(Churn ~ ., data = smote_train[, c(full_features, "Churn")], family = binomial(link = "logit"))
logit_top <- glm(Churn ~ ., data = smote_train[, c(top_features, "Churn")], family = binomial(link = "logit"))

# Random Forest
cat("Training Random Forest on full and top 10 feature sets...\n")
rf_full <- train(
  Churn ~ .,
  data = smote_train[, c(full_features, "Churn")],
  method = "rf",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 100
)
rf_top <- train(
  Churn ~ .,
  data = smote_train[, c(top_features, "Churn")],
  method = "rf",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  ntree = 100
)

# XGBoost
cat("Training XGBoost on full and top 10 feature sets...\n")
xgb_full <- train(
  Churn ~ .,
  data = smote_train[, c(full_features, "Churn")],
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE
)
xgb_top <- train(
  Churn ~ .,
  data = smote_train[, c(top_features, "Churn")],
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  verbose = FALSE
)

# 4. Evaluate models on test set using AUC
# Predict probabilities
logit_full_probs <- predict(logit_full, newdata = test_data, type = "response")
logit_top_probs <- predict(logit_top, newdata = test_data, type = "response")
rf_full_probs <- predict(rf_full, newdata = test_data, type = "prob")[, "Yes"]
rf_top_probs <- predict(rf_top, newdata = test_data, type = "prob")[, "Yes"]
xgb_full_probs <- predict(xgb_full, newdata = test_data, type = "prob")[, "Yes"]
xgb_top_probs <- predict(xgb_top, newdata = test_data, type = "prob")[, "Yes"]

# Calculate AUC
logit_full_roc <- roc(test_data$Churn, logit_full_probs, levels = c("No", "Yes"), direction = "<")
logit_top_roc <- roc(test_data$Churn, logit_top_probs, levels = c("No", "Yes"), direction = "<")
rf_full_roc <- roc(test_data$Churn, rf_full_probs, levels = c("No", "Yes"), direction = "<")
rf_top_roc <- roc(test_data$Churn, rf_top_probs, levels = c("No", "Yes"), direction = "<")
xgb_full_roc <- roc(test_data$Churn, xgb_full_probs, levels = c("No", "Yes"), direction = "<")
xgb_top_roc <- roc(test_data$Churn, xgb_top_probs, levels = c("No", "Yes"), direction = "<")

# 5. Summarize AUC results
feature_set_performance <- data.frame(
  Model = rep(c("Logistic Regression", "Random Forest", "XGBoost"), each = 2),
  Feature_Set = rep(c("Full Features", "Top 10 Features"), times = 3),
  AUC = round(c(
    auc(logit_full_roc), auc(logit_top_roc),
    auc(rf_full_roc), auc(rf_top_roc),
    auc(xgb_full_roc), auc(xgb_top_roc)
  ), 3)
)

# Display the summary table
cat("\nAUC Comparison Across Models and Feature Sets:\n")
kable(feature_set_performance, caption = "AUC Performance for Different Models and Feature Sets")

# 6.1 Visualize ROC curves for all models and feature sets
roc_data_features <- data.frame(
  Model_Feature = c(
    rep("Logistic Regression (Full)", length(logit_full_roc$sensitivities)),
    rep("Logistic Regression (Top 10)", length(logit_top_roc$sensitivities)),
    rep("Random Forest (Full)", length(rf_full_roc$sensitivities)),
    rep("Random Forest (Top 10)", length(rf_top_roc$sensitivities)),
    rep("XGBoost (Full)", length(xgb_full_roc$sensitivities)),
    rep("XGBoost (Top 10)", length(xgb_top_roc$sensitivities))
  ),
  Sensitivity = c(
    logit_full_roc$sensitivities, logit_top_roc$sensitivities,
    rf_full_roc$sensitivities, rf_top_roc$sensitivities,
    xgb_full_roc$sensitivities, xgb_top_roc$sensitivities
  ),
  Specificity = c(
    1 - logit_full_roc$specificities, 1 - logit_top_roc$specificities,
    1 - rf_full_roc$specificities, 1 - rf_top_roc$specificities,
    1 - xgb_full_roc$specificities, 1 - xgb_top_roc$specificities
  )
)

p_roc_features <- ggplot(roc_data_features, aes(x = Specificity, y = Sensitivity, color = Model_Feature, group = Model_Feature)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray70") +
  scale_color_manual(values = c(
    "Logistic Regression (Full)" = "#9b1c31",  
    "Logistic Regression (Top 10)" = "#eb9191",  
    "Random Forest (Full)" = "#388e3c",  
    "Random Forest (Top 10)" = "#a0dea2",  
    "XGBoost (Full)" = "#8e44ad",  
    "XGBoost (Top 10)" = "#ddb6e3"  
  )) +
  labs(
    title = "ROC Curves for Models with Different Feature Sets",
    subtitle = "Comparing Full Features vs. Top 10 Features Across Models",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    caption = paste(
      "AUC: Logistic Regression (Full) =", round(auc(logit_full_roc), 3),
      "| Logistic Regression (Top 10) =", round(auc(logit_top_roc), 3),
      "| Random Forest (Full) =", round(auc(rf_full_roc), 3),
      "| Random Forest (Top 10) =", round(auc(rf_top_roc), 3),
      "| XGBoost (Full) =", round(auc(xgb_full_roc), 3),
      "| XGBoost (Top 10) =", round(auc(xgb_top_roc), 3)
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 12),
    plot.caption = element_text(hjust = 1, face = "italic", size = 10, color = "gray50"),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10),
    plot.margin = ggplot2::margin(t = 15, r = 25, b = 15, l = 25)
  )

# Print the ROC plot
print(p_roc_features)

# 6.2 Visualize AUC Differences with a Bar Plot
auc_plot_data <- feature_set_performance
auc_plot_data$Model_Feature <- paste(auc_plot_data$Model, auc_plot_data$Feature_Set, sep = " - ")

p_auc_bar <- ggplot(auc_plot_data, aes(x = reorder(Model_Feature, AUC), y = AUC, fill = Feature_Set)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = AUC), position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Full Features" = "#264653", "Top 10 Features" = "#E76F51")) +
  labs(
    title = "AUC Comparison Across Models and Feature Sets",
    subtitle = "Highlighting Performance Differences",
    x = "Model and Feature Set",
    y = "AUC",
    fill = "Feature Set"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50", size = 12),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    plot.margin = ggplot2::margin(t = 15, r = 25, b = 15, l = 25)
  )

# Print the bar plot
print(p_auc_bar)
ggsave("outputs/auc_feature_sets.png", p_auc_bar, width = 10, height = 7, dpi = 150, bg = "white")
