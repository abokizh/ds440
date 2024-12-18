import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

class Model:
    
    def __init__(self):
        
        # Already did Hyperparameter Tuning
        # Loading Data
        self.file_path = '40kexpandedEMSData_withFakeData.csv'
        self.data = pd.read_csv(self.file_path)

        # Target variable is ETA Prediction in minutes
        self.y = self.data['response_time_minutes']

        # Features while Dropping columns that aren't useful for the prediction
        self.X = self.data.drop(columns=['cad_incident_id', 'incident_datetime', 'response_time_minutes', 'first_assignment_datetime', 
                            'first_activation_datetime', 'first_hosp_arrival_datetime', 'incident_close_datetime'])

        # One Hot Encoding for categorical features into numerical variables
        self.categorical_cols = ['initial_call_type', 'final_call_type', 'borough', 'weather_condition', 'day_of_week', 'time_of_day']

        # Numerical columns (already in numeric form)
        self.numerical_cols = ['initial_severity_level_code', 'final_severity_level_code', 'incident_hour', 'is_weekend', 
                        'temperature', 'traffic_severity', 'is_peak_hour', 'borough_density', 'is_holiday', 'distance_estimate']

        # Preprocessing Pipeline with Imputation
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                        ('scaler', StandardScaler())]), self.numerical_cols),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))]), self.categorical_cols)
            ])

        
    def train(self):

        # Hardcoded Best XGBoost Parameters
        self.xgb = XGBRegressor(
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
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('regressor', self.xgb)])

        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Fit the model
        self.pipeline.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred_train = self.pipeline.predict(self.X_train)
        self.y_pred_test = self.pipeline.predict(self.X_test)

        # Evaluate the model (Mean Squared Error)
        self.train_mse = mean_squared_error(self.y_train, self.y_pred_train)
        self.test_mse = mean_squared_error(self.y_test, self.y_pred_test)

    def predict(self, type, borough, zip, temperature, weather_condition, traffic_severity):
        user_input = {
            'initial_call_type': [type],
            'initial_severity_level_code': [5],
            'final_call_type': [type],
            'final_severity_level_code': [5],

            'valid_dispatch_rspns_time_indc': ['Y'],
            'dispatch_response_seconds_qy': [167],
            'first_on_scene_datetime': ['2024-01-01T00:09:34.000'],
            'valid_incident_rspns_time_indc': ['Y'],
            'incident_response_seconds_qy': [587],
            'incident_travel_tm_seconds_qy': [420],
            'first_to_hosp_datetime': ['2024-01-01T00:33:26.000'],
            'borough': [borough],
            'zipcode': [zip],
            'policeprecinct': [112],
            'citycouncildistrict': [29],
            'communitydistrict': [406],

            'communityschooldistrict': [28],
            'congressionaldistrict': [6],
            'incident_hour': [23],
            'time_of_day': ['night'],
            'day_of_week': ['Sunday'],
            'is_weekend': [1],
            'severity_change': [0],
            'weather_condition': [weather_condition],
            'temperature': [temperature],
            'is_peak_hour': [0],
            'incident_duration_minutes': [48.6],
            'traffic_severity': [traffic_severity],

            'borough_density': [21000],
            'is_holiday': [1],
            'distance_estimate': [9.5],
        }
        user_input_df = pd.DataFrame(user_input)
        self.prediction = self.pipeline.predict(user_input_df)
        return self.prediction


