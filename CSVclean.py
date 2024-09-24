import pandas as pd
import os


# Set relative path for the project directory
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'small data NYC EMS.xlsx')

# Load the Excel file and convert it to CSV
data = pd.read_excel(file_path)

# Save as CSV for future use
csv_file_path = os.path.join(current_dir, 'small_data_NYC_EMS.csv')
data.to_csv(csv_file_path, index=False)

# Read the newly saved CSV file
# Use encoding to prevent Unicode errors
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Convert time columns to datetime
data['incident_datetime'] = pd.to_datetime(data['incident_datetime'], errors='coerce')
data['first_assignment_datetime'] = pd.to_datetime(data['first_assignment_datetime'], errors='coerce')
data['first_activation_datetime'] = pd.to_datetime(data['first_activation_datetime'], errors='coerce')

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

# Handle missing values by dropping rows with critical missing data
data = data.dropna(subset=['incident_datetime', 'first_assignment_datetime', 'first_activation_datetime'])

# Save the cleaned data to a new CSV
cleaned_file_path = os.path.join(current_dir, 'cleaned_ems_data.csv')
data.to_csv(cleaned_file_path, index=False)

cleaned_file_path
