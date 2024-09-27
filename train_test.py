import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer  
import matplotlib.pyplot as plt

# Loading Data
file_path = 'newCSVWith_Weather_Trafficfeatures.csv'
data = pd.read_csv(file_path)

# Target variable is ETA Prediction in minutes
y = data['response_time_minutes']

# Features while Dropping columns that aren't useful for the prediction
X = data.drop(columns=['cad_incident_id', 'incident_datetime', 'response_time_minutes', 'first_assignment_datetime', 
                       'first_activation_datetime', 'first_hosp_arrival_datetime', 'incident_close_datetime'])

# One Hot Encoding for categorical features into numerical variables
categorical_cols = ['initial_call_type', 'final_call_type', 'borough', 'weather_condition', 'day_of_week', 'time_of_day']

# Numerical columns (already in numeric form)
numerical_cols = ['initial_severity_level_code', 'final_severity_level_code', 'incident_hour', 'is_weekend', 
                  'temperature', 'is_peak_hour', 'traffic_severity', 'borough_density', 'is_holiday', 'distance_estimate']

# Preprocessing Pipeline with Imputation
preprocessor = ColumnTransformer(
    transformers=[
        # For numerical columns, impute missing values with the mean and then scale
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                ('scaler', StandardScaler())]), numerical_cols),
        # For categorical columns, impute missing values with the most frequent value and then one-hot encode
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])


# Model pipeline (preprocessing + Random Forest Regressor)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model (Mean Squared Error)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")


# Calculate RMSE
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# ----------- Feature Importance ---------------

# Extract the random forest regressor from the pipeline
rf_model = model.named_steps['regressor']

# Extract the feature importances
importances = rf_model.feature_importances_

# Get the column names after preprocessing
cat_features = model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numerical_cols, cat_features])

# Sort the feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = all_features[sorted_indices]

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.title("Feature Importance")
plt.barh(sorted_features, sorted_importances)
plt.xlabel("Feature Importance Score")
plt.gca().invert_yaxis()  # Highest importance on top

# Adjust font size and rotation for readability
plt.xticks(fontsize=10)
plt.yticks(fontsize=4)  # Reduce font size for better readability

# Optionally limit the number of features displayed (e.g., top 20 most important)
plt.barh(sorted_features[:20], sorted_importances[:20])

plt.show()