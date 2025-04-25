import numpy as np
import pandas as pd

# Simulate hourly temperature for January in a specific location
def simulate_temperature(hour, day_of_year):
    # Base temperature (e.g., average for January)
    base_temp = 5  # Adjust for your location
    # Daily variation (sinusoidal pattern)
    daily_variation = 10 * np.sin(np.pi * (hour - 6) / 12)  # Peaks at 3 PM
    # Seasonal variation (e.g., colder in January)
    seasonal_variation = -5 * np.cos(2 * np.pi * day_of_year / 365)
    # Add randomness
    random_noise = np.random.normal(0, 2)
    return base_temp + daily_variation + seasonal_variation + random_noise

# Simulate precipitation probability
def simulate_precipitation(hour):
    # Higher probability in the evening
    base_probability = 0.1 + 0.15 * np.sin(np.pi * (hour - 15) / 12)
    return np.random.random() < base_probability

# Generate data for January
dates = pd.date_range('2025-01-01', '2025-01-31', freq='h')
weather_data = []
for date in dates:
    temp = simulate_temperature(date.hour, date.dayofyear)
    precip = simulate_precipitation(date.hour)
    weather_data.append({'date': date, 'hour': date.hour, 'temp': temp, 'precip': precip})

# Create DataFrame
weather_df = pd.DataFrame(weather_data)
print(weather_df.head())