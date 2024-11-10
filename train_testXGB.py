import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Already did Hyperparameter Tuning
# Loading Data
file_path = '40kexpandedEMSData_withFakeData.csv'
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
                  'temperature', 'traffic_severity', 'is_peak_hour', 'borough_density', 'is_holiday', 'distance_estimate']

# Preprocessing Pipeline with Imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Hardcoded Best XGBoost Parameters
xgb = XGBRegressor(
    random_state=42,
    colsample_bytree=1.0,
    gamma=0.3,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=10,
    n_estimators=200,
    subsample=1.0,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0  # L2 regularization
)


# Model pipeline (preprocessing + XGBoost Regressor)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', xgb)])

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

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