import pandas as pd
import numpy as np
from datetime import timedelta

# Load the original data
file_path = 'small data NYC EMS.xlsx'  # Update with relative path if needed
data = pd.read_excel(file_path, engine='openpyxl')

# Define the date range for randomization
start_date = pd.Timestamp('2024-01-02')
end_date = pd.Timestamp('2024-12-31')

# Function to generate random dates within the range
def random_dates(start, end, num=1):
    start_u = start.value//10**9
    end_u = end.value//10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, num), unit='s')

# Creating 10,000 additional rows by sampling and modifying existing data
new_rows = data.sample(n=10000, replace=True).reset_index(drop=True)

# Randomize the 'incident_datetime' within the specified range
new_rows['incident_datetime'] = random_dates(start_date, end_date, num=10000)

# Randomize related time columns based on new 'incident_datetime'
time_deltas = pd.to_timedelta(np.random.randint(5, 3600, size=10000), unit='s')  # Random time deltas between 5s to 1hr
new_rows['first_assignment_datetime'] = new_rows['incident_datetime'] + time_deltas
new_rows['first_activation_datetime'] = new_rows['first_assignment_datetime'] + pd.to_timedelta(np.random.randint(60, 600, size=10000), unit='s')

# Randomize 'response_time_minutes'
new_rows['response_time_minutes'] = (new_rows['first_assignment_datetime'] - new_rows['incident_datetime']).dt.total_seconds() / 60

# Optional: Add more randomness to other columns if necessary
# Example: Randomize 'initial_severity_level_code' within a range
severity_levels = data['initial_severity_level_code'].unique()
new_rows['initial_severity_level_code'] = np.random.choice(severity_levels, size=10000)

# Combine the original data with the new fake data
combined_data = pd.concat([data, new_rows], ignore_index=True)

# Save the combined dataset to a new CSV file
combined_file_path = 'expanded_ems_data.csv'
combined_data.to_csv(combined_file_path, index=False)

# Show some of the new data
combined_data.tail()
