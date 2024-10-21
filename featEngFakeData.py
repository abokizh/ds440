import pandas as pd
import os
import numpy as np
from datetime import timedelta

# Set relative path for the project directory
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'small data NYC EMS.xlsx')

# Load the Excel file and convert it to CSV
data = pd.read_excel(file_path)

# Save as CSV for future use
csv_file_path = os.path.join(current_dir, 'small_data_NYC_EMS.csv')
data.to_csv(csv_file_path, index=False)

# Read the newly saved CSV file
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Convert time columns to datetime
data['incident_datetime'] = pd.to_datetime(data['incident_datetime'], errors='coerce')
data['first_assignment_datetime'] = pd.to_datetime(data['first_assignment_datetime'], errors='coerce')
data['first_activation_datetime'] = pd.to_datetime(data['first_activation_datetime'], errors='coerce')
data['first_hosp_arrival_datetime'] = pd.to_datetime(data['first_hosp_arrival_datetime'], errors='coerce')

# Drop irrelevant columns
irrelevant_columns = ['reopen_indicator', 'special_event_indicator', 'standby_indicator', 'transfer_indicator']
data = data.drop(columns=irrelevant_columns)

# Feature 1: Calculate response time (in minutes)
data['response_time_minutes'] = (data['first_assignment_datetime'] - data['incident_datetime']).dt.total_seconds() / 60

# Feature 2: Time of Day
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

data['incident_hour'] = data['incident_datetime'].dt.hour
data['time_of_day'] = data['incident_hour'].apply(time_of_day)

# Feature 3: Day of the Week
data['day_of_week'] = data['incident_datetime'].dt.day_name()

# Feature 4: Weekday or Weekend
data['is_weekend'] = data['incident_datetime'].dt.weekday >= 5

# Feature 5: Severity difference
data['severity_change'] = data['initial_severity_level_code'] - data['final_severity_level_code']

# Adding Weather Data Columns (Seasonal Weather Conditions)
weather_conditions_winter = ['Snow', 'Rainy', 'Cloudy', 'Windy', 'Foggy']
data['weather_condition'] = np.random.choice(weather_conditions_winter, size=len(data), p=[0.4, 0.2, 0.3, 0.05, 0.05])

# Average Temperature based on Borough
borough_avg_temp = {
    'MANHATTAN': 35,
    'BROOKLYN': 34,
    'QUEENS': 33,
    'BRONX': 32,
    'STATEN ISLAND': 31
}
data['temperature'] = data['borough'].apply(lambda x: borough_avg_temp.get(x, 33) + np.random.randint(-5, 5))

# Feature 6: Peak Hour (Rush hour is defined as 7-9 AM and 4-7 PM)
def is_peak_hour(hour):
    return 1 if (7 <= hour < 9) or (16 <= hour < 19) else 0

data['is_peak_hour'] = data['incident_hour'].apply(is_peak_hour)

# Feature 7: Incident Duration (Minutes from incident to hospital arrival)
data['incident_duration_minutes'] = (data['first_hosp_arrival_datetime'] - data['incident_datetime']).dt.total_seconds() / 60

# Feature 8: Traffic Severity (Simulated score from 0-10 based on time of day)
def traffic_severity(hour):
    if 7 <= hour < 9 or 16 <= hour < 19:  # Defined as rush hours
        return np.random.uniform(7, 10)
    elif 9 <= hour < 16:  # Middle of the day Traffic Severity
        return np.random.uniform(4, 6)
    else:  # Early morning and late night Traffic Severity
        return np.random.uniform(0, 3)

data['traffic_severity'] = data['incident_hour'].apply(traffic_severity)

# Feature 9: Borough Density (Simulated population density values for each borough) with people per square mile
borough_density = {
    'MANHATTAN': 70000, 
    'BROOKLYN': 36000,
    'QUEENS': 21000,
    'BRONX': 34000,
    'STATEN ISLAND': 8200
}
data['borough_density'] = data['borough'].map(borough_density)

# Feature 10: Holiday or not? (Indicating if the incident occurred on New Year's Day, Christmas Day, or New Year's Eve)
us_holidays = pd.to_datetime(['2024-01-01', '2023-12-25', '2023-12-31'])  
data['is_holiday'] = data['incident_datetime'].dt.date.isin(us_holidays.date)

# Feature 11: Distance Estimate (Simulated based on borough and severity)
# This feature can be improved with actual data later
def estimate_distance(borough, severity):
    # Higher severity and borough size affect the distance estimate
    base_distance = {'MANHATTAN': 2, 'BROOKLYN': 5, 'QUEENS': 7, 'BRONX': 4, 'STATEN ISLAND': 8}.get(borough, 5)
    return base_distance + (severity * 0.5)  # More severe incidents tend to travel longer distances

data['distance_estimate'] = data.apply(lambda row: estimate_distance(row['borough'], row['initial_severity_level_code']), axis=1)

# Handle missing values by dropping rows with critical missing data
data = data.dropna(subset=['incident_datetime', 'first_assignment_datetime', 'first_activation_datetime'])

# Function to generate random dates within the range
def random_dates(start, end, num=1):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, num), unit='s')

# Adding 10,000 more fake rows
start_date = pd.Timestamp('2024-01-02')
end_date = pd.Timestamp('2024-12-31')
new_rows = data.sample(n=10000, replace=True).reset_index(drop=True)

# Randomize the 'incident_datetime' within the specified range
new_rows['incident_datetime'] = random_dates(start_date, end_date, num=10000)

# Randomize related time columns based on new 'incident_datetime'
time_deltas = pd.to_timedelta(np.random.randint(5, 3600, size=10000), unit='s')  # Random time deltas between 5s to 1hr
new_rows['first_assignment_datetime'] = new_rows['incident_datetime'] + time_deltas
new_rows['first_activation_datetime'] = new_rows['first_assignment_datetime'] + pd.to_timedelta(np.random.randint(60, 600, size=10000), unit='s')
new_rows['first_hosp_arrival_datetime'] = new_rows['first_activation_datetime'] + pd.to_timedelta(np.random.randint(300, 1800, size=10000), unit='s')

# Combine the original data with the new fake data
combined_data = pd.concat([data, new_rows], ignore_index=True)

# Save the cleaned and enhanced data to a new CSV with added features and fake rows
cleaned_file_path = os.path.join(current_dir, 'newCSVWith_FakeData.csv')
combined_data.to_csv(cleaned_file_path, index=False)

# Output the new CSV file path for reference
cleaned_file_path
