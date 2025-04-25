import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.weatherData import weather_df

  # Call the function to generate weather data
# Load the cleaned coffee shop data
df_full = pd.read_excel('data/cleaned_coffee_shop_data.xlsx')

# Create a date column with all days in January for each hour entry
dates = pd.date_range('2025-01-01', '2025-01-31')
df_full = df_full.loc[df_full.index.repeat(len(dates))].reset_index(drop=True)
df_full['date'] = dates.repeat(len(df_full) // len(dates))

# Convert dates to datetime format in both DataFrames
df_full['date'] = pd.to_datetime(df_full['date'])
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Create date string columns for merging
df_full['date_str'] = df_full['date'].dt.date.astype(str)
weather_df['date_str'] = weather_df['date'].dt.date.astype(str)

# Merge the dataframes on date_str and hour
merged_df = pd.merge(df_full, weather_df,
                     on=['date_str', 'hour'],
                     how='left',
                     suffixes=('', '_weather'))

# Clean up duplicate columns
merged_df = merged_df.drop(['date_weather'], axis=1)
merged_df = merged_df.rename(columns={'temp': 'temperature', 'precip': 'precipitation'})

# Create flag for weekends
merged_df['is_weekend'] = merged_df['date'].dt.dayofweek >= 5

# Create weather effects on coffee sales (simulate realistic relationships)
# People drink more coffee when it's cold, less when it's very hot
merged_df['baseline_checks'] = merged_df['check_count'].copy()

# Temperature effect: more coffee when cold, less when hot
merged_df['temp_effect'] = np.where(
    merged_df['temperature'] < -5, 1.3,  # Very cold: +30% sales
    np.where(
        merged_df['temperature'] < 5, 1.15,  # Cold: +15% sales
        np.where(
            merged_df['temperature'] > 20, 0.85,  # Hot: -15% sales
            1.0  # Normal temperatures: no effect
        )
    )
)

# Precipitation effect: more coffee when it's raining
merged_df['precip_effect'] = np.where(
    merged_df['precipitation'], 1.1, 1.0  # +10% sales when precipitating
)

# Apply the weather effects (with 20% influence)
weather_factor = 0.2
merged_df['check_count'] = merged_df['baseline_checks'] * (
    (1 - weather_factor) +
    weather_factor * merged_df['temp_effect'] * merged_df['precip_effect']
)
merged_df['check_count'] = merged_df['check_count'].round().astype(int)

# --- Weekday-Hour Heatmap ---
merged_df['weekday'] = merged_df['date'].dt.day_name()
pivot = merged_df.pivot_table(
    index='weekday', columns='hour', values='check_count', aggfunc='mean'
)

# Custom weekday order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(weekday_order)

plt.figure(figsize=(12, 6))
heatmap = plt.imshow(pivot, aspect='auto', cmap='viridis')
plt.colorbar(heatmap, label='Average Check Count')
plt.title('Coffee Shop Traffic by Weekday and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.xticks(range(pivot.shape[1]), range(pivot.shape[1]))
plt.yticks(range(pivot.shape[0]), pivot.index)
plt.tight_layout()

# Save the figure
plt.savefig('weekday_hour_heatmap.png', dpi=300, bbox_inches='tight')

# Optionally show it if running in an interactive environment
plt.show()

# --- Additional Analysis: Peak Hours ---
# Identify peak hours by day
peak_hours = pivot.idxmax(axis=1)
print("\nPeak Hours by Day:")
for day, hour in peak_hours.items():
    print(f"{day}: {hour}:00")

# --- Temperature Correlation Analysis ---
# Calculate correlation between temperature and check count for each hour
hour_temp_correlations = []

for hour in range(24):
    hour_data = merged_df[merged_df['hour'] == hour]
    if len(hour_data) > 5:  # Need enough data points
        temp_corr = hour_data['temperature'].corr(hour_data['check_count'])
        hour_temp_correlations.append(temp_corr)
    else:
        hour_temp_correlations.append(np.nan)

# Plot temperature correlations
plt.figure(figsize=(10, 5))
plt.bar(range(24), hour_temp_correlations)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Correlation Between Temperature and Coffee Shop Traffic by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Correlation Coefficient')
plt.xticks(range(0, 24, 2))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('temp_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Precipitation Correlation Analysis ---
# Calculate correlation between precipitation and check count for each hour
hour_precip_correlations = []

for hour in range(24):
    hour_data = merged_df[merged_df['hour'] == hour]
    if len(hour_data) > 5:
        # For precipitation (which is boolean), use point-biserial correlation
        # This is automatically handled by pandas corr()
        precip_corr = hour_data['precipitation'].corr(hour_data['check_count'])
        hour_precip_correlations.append(precip_corr)
    else:
        hour_precip_correlations.append(np.nan)

# Plot precipitation correlations
plt.figure(figsize=(10, 5))
plt.bar(range(24), hour_precip_correlations)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Correlation Between Precipitation and Coffee Shop Traffic by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Correlation Coefficient')
plt.xticks(range(0, 24, 2))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('precip_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Scatter plot of temperature vs. check count ---
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['temperature'], merged_df['check_count'], alpha=0.3)
plt.title('Temperature vs Coffee Shop Traffic')
plt.xlabel('Temperature (°C)')
plt.ylabel('Check Count')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('temp_scatter.png', dpi=300, bbox_inches='tight')
plt.show()