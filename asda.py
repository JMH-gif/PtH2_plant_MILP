import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import (num_representative_weeks, plot_average_week, plot_parameter, representative_week_method)

# Load the dataset and create a datetime index
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        start_datetime = '2021-01-01 00:00'  # Adjust as needed
        df.index = pd.date_range(start=start_datetime, periods=len(df), freq='h')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_representative_weeks(data):
    hours_per_week = 24 * 7

    # Define seasons using datetime ranges
    spring_start, spring_end = '2021-03-01 00:00', '2021-05-31 23:00'
    summer_start, summer_end = '2021-06-01 00:00', '2021-08-31 23:00'
    autumn_start, autumn_end = '2021-09-01 00:00', '2021-11-30 23:00'
    winter_start1, winter_end1 = '2021-01-01 00:00', '2021-02-28 23:00'
    winter_start2, winter_end2 = '2021-12-01 00:00', '2021-12-31 23:00'

    # Extract data for each season based on these datetime ranges
    spring_data = data[spring_start:spring_end]
    summer_data = data[summer_start:summer_end]
    autumn_data = data[autumn_start:autumn_end]
    winter_data = pd.concat([data[winter_start1:winter_end1], data[winter_start2:winter_end2]])

    # Function to calculate an average week across all data points for each hour in a season
    def average_week_per_hour(season_data):
        season_data = season_data.copy()
        season_data['hour_of_week'] = season_data.index.weekday * 24 + season_data.index.hour
        averaged_week = season_data.groupby('hour_of_week').mean(numeric_only=True)
        return averaged_week

    if representative_week_method == "random":
        def select_full_weeks(season_data, num_weeks):
            season_data = season_data.copy()
            season_data['week_number'] = season_data.index.isocalendar().week
            unique_weeks = season_data['week_number'].unique()
            selected_weeks = np.random.choice(unique_weeks, num_weeks, replace=False)
            selected_data = season_data[season_data['week_number'].isin(selected_weeks)]
            return selected_data.drop(columns='week_number')

        condensed_spring = select_full_weeks(spring_data, num_representative_weeks)
        condensed_summer = select_full_weeks(summer_data, num_representative_weeks)
        condensed_autumn = select_full_weeks(autumn_data, num_representative_weeks)
        condensed_winter = select_full_weeks(winter_data, num_representative_weeks)

    elif representative_week_method == "average":
        condensed_spring = average_week_per_hour(spring_data)
        condensed_summer = average_week_per_hour(summer_data)
        condensed_autumn = average_week_per_hour(autumn_data)
        condensed_winter = average_week_per_hour(winter_data)

        if plot_average_week:
            plot_seasonal_averages(
                [spring_data, summer_data, autumn_data, winter_data],
                [condensed_spring, condensed_summer, condensed_autumn, condensed_winter],
                plot_parameter
            )

    # Concatenate the representative weeks in seasonal order and reset index for modeling use
    condensed_year = pd.concat([condensed_spring, condensed_summer, condensed_autumn, condensed_winter])
    condensed_year = condensed_year.reset_index(drop=True)

    return condensed_year

def plot_seasonal_averages(season_data_list, averaged_weeks_list, parameter):
    sns.set(style="darkgrid")
    seasons = ["Spring", "Summer", "Autumn", "Winter"]

    y_label = {
        "el_price": "Electricity Price [€/MWh]",
        "cf_wind": "Wind Capacity Factor",
        "cf_solar": "Solar Capacity Factor"
    }.get(parameter, "Value")

    for i, (season_data, avg_week) in enumerate(zip(season_data_list, averaged_weeks_list)):
        season_data = season_data.copy()
        season_data['hour_of_week'] = season_data.index.weekday * 24 + season_data.index.hour

        hourly_values = season_data.groupby('hour_of_week')[parameter].apply(list)

        plt.figure(figsize=(14, 6))
        for hour, values in hourly_values.items():
            plt.scatter([hour] * len(values), values, color="blue", alpha=0.3)
        plt.plot(avg_week.index, avg_week[parameter], color="red", linewidth=2, label="Average Week")

        plt.title(f"{seasons[i]} - Average Week for {parameter}")
        plt.xlabel("Hour of the Week")
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        plt.show()
# Function to return the condensed data for model use
def get_condensed_data(file_path):
    data = load_data(file_path)
    if data is not None:
        return create_representative_weeks(data)
    else:
        print("Failed to load data.")
        return None

if __name__ == "__main__":
    file_path = r"C:\Users\Mika\Desktop\Master\Data\Hourly_el_price_cf_wind_cf_solar.xlsx"

    data = load_data(file_path)
    if data is not None:
        condensed_data = create_representative_weeks(data)

        # Prepare data for the model
        time_varying_scenario = {
            'cf_wind': condensed_data['cf_wind'].tolist(),
            'cf_solar': condensed_data['cf_solar'].tolist(),
            'el_price': condensed_data['el_price'].tolist(),
        }
        print(time_varying_scenario)
    else:
         print("Failed to load data. Please check the file path and data format.")
