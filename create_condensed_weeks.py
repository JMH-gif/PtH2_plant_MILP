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

    # Function to calculate an average week and dynamic bounds (min, max) for each hour
    def average_week_per_hour_with_bounds(season_data):
        season_data = season_data.copy()
        season_data['hour_of_week'] = season_data.index.weekday * 24 + season_data.index.hour

        # Calculate mean, min, and max values for each hour
        averaged_week = season_data.groupby('hour_of_week').mean(numeric_only=True)
        min_values = season_data.groupby('hour_of_week').min(numeric_only=True)
        max_values = season_data.groupby('hour_of_week').max(numeric_only=True)

        bounds = pd.DataFrame({
            ('el_price', 'min'): min_values['el_price'],
            ('el_price', 'max'): max_values['el_price'],
            ('cf_wind', 'min'): min_values['cf_wind'],
            ('cf_wind', 'max'): max_values['cf_wind'],
            ('cf_solar', 'min'): min_values['cf_solar'],
            ('cf_solar', 'max'): max_values['cf_solar']
        })

        bounds.columns = pd.MultiIndex.from_tuples(bounds.columns)


        return averaged_week, bounds

    if representative_week_method == "random":
        def select_full_weeks(season_data, num_weeks):
            """
            Selects a specified number of random full weeks from the given seasonal data.
            Ensures each selected week has exactly 168 hours.
            """
            season_data = season_data.copy()
            season_data['week_number'] = season_data.index.isocalendar().week
            unique_weeks = season_data['week_number'].unique()
            selected_weeks = []

            # Iterate until we have the required number of valid weeks
            while len(selected_weeks) < num_weeks:
                week_to_check = np.random.choice(unique_weeks, 1, replace=False)[0]
                week_data = season_data[season_data['week_number'] == week_to_check]

                # Check if the week has exactly 168 rows
                if len(week_data) == 168:
                    selected_weeks.append(week_data)
                    unique_weeks = unique_weeks[unique_weeks != week_to_check]  # Remove selected week

            # Combine selected weeks into a single DataFrame
            selected_data = pd.concat(selected_weeks)
            selected_data = selected_data.drop(columns='week_number')
            return selected_data

        condensed_spring = select_full_weeks(spring_data, num_representative_weeks)
        condensed_summer = select_full_weeks(summer_data, num_representative_weeks)
        condensed_autumn = select_full_weeks(autumn_data, num_representative_weeks)
        condensed_winter = select_full_weeks(winter_data, num_representative_weeks)

        # Initialize bounds variables as empty DataFrames
        spring_bounds = pd.DataFrame()
        summer_bounds = pd.DataFrame()
        autumn_bounds = pd.DataFrame()
        winter_bounds = pd.DataFrame()

    elif representative_week_method == "average":
        condensed_spring, spring_bounds = average_week_per_hour_with_bounds(spring_data)
        condensed_summer, summer_bounds = average_week_per_hour_with_bounds(summer_data)
        condensed_autumn, autumn_bounds = average_week_per_hour_with_bounds(autumn_data)
        condensed_winter, winter_bounds = average_week_per_hour_with_bounds(winter_data)

        if plot_average_week:
            plot_seasonal_averages(
                [spring_data, summer_data, autumn_data, winter_data],
                [condensed_spring, condensed_summer, condensed_autumn, condensed_winter],
                plot_parameter,
                [spring_bounds, summer_bounds, autumn_bounds, winter_bounds],
                [spring_bounds, summer_bounds, autumn_bounds, winter_bounds],
                plot_bounds=True  # Use the toggle from config
            )

    # Concatenate the representative weeks in seasonal order and reset index for modeling use
    condensed_year = pd.concat([condensed_spring, condensed_summer, condensed_autumn, condensed_winter])
    bounds_year = pd.concat([spring_bounds, summer_bounds, autumn_bounds, winter_bounds])

    condensed_year = condensed_year.reset_index(drop=True)
    bounds_year = bounds_year.reset_index(drop=True)

    # Save bounds to CSV with hierarchical indexing
    bounds_year.to_csv("multi_index_bounds.csv")

    def plot_seasonal_duration_curves(season_data, avg_week_data, parameter, season_name):
        """
        Compares the duration curve for the average week per season with the duration curve for the entire season.

        Parameters:
        - season_data (pd.DataFrame): Full season data with datetime index.
        - avg_week_data (pd.DataFrame): Average week data with hourly index.
        - parameter (str): The parameter to plot (e.g., 'el_price', 'cf_wind', 'cf_solar').
        - season_name (str): Name of the season for labeling purposes.
        """
        # Sort values for the entire season (duration curve)
        season_sorted = season_data[parameter].sort_values(ascending=False).values

        # Sort values for the average week (duration curve)
        avg_week_sorted = avg_week_data[parameter].sort_values(ascending=False).values

        # Plot duration curves
        plt.figure(figsize=(12, 6))
        plt.plot(season_sorted, label=f'{season_name} - Entire Season', linewidth=2)
        plt.plot(avg_week_sorted, label=f'{season_name} - Average Week', linewidth=2, linestyle='--')

        # Add labels, title, and legend
        plt.title(f'Duration Curve Comparison for {parameter} ({season_name})', fontsize=16)
        plt.xlabel('Hours', fontsize=14)
        plt.ylabel(parameter, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Example for Spring
        plot_seasonal_duration_curves(
            spring_data,  # Entire season data
            condensed_spring,  # Average week data
            parameter='el_price',  # Parameter to plot
            season_name='Spring'  # Name of the season
        )

    return condensed_year, bounds_year


import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_seasonal_averages(season_data_list, averaged_weeks_list, parameter, max_bounds, min_bounds, plot_bounds=False):
    sns.set(style="darkgrid")
    seasons = ["Spring", "Summer", "Autumn", "Winter"]

    # Dynamic y-axis label based on parameter
    y_label = {
        "el_price": "Electricity Price [€/MWh]",
        "cf_wind": "Wind Capacity Factor",
        "cf_solar": "Solar Capacity Factor"
    }.get(parameter, "Value")

    # Save directory path
    save_dir = r"C:\Users\Mika\Desktop\Master\Pictures_Master"

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through seasons and plot
    for i, (season_data, avg_week, max_bound, min_bound) in enumerate(zip(season_data_list, averaged_weeks_list, max_bounds, min_bounds)):
        season_data = season_data.copy()
        season_data['hour_of_week'] = season_data.index.weekday * 24 + season_data.index.hour

        hourly_values = season_data.groupby('hour_of_week')[parameter].apply(list)

        # Figure size
        plt.figure(figsize=(14, 6))

        # Scatter plot for raw data
        for hour, values in hourly_values.items():
            plt.scatter([hour] * len(values), values, color="blue", alpha=0.3)

        # Line plot for average week
        plt.plot(avg_week.index, avg_week[parameter], color="red", linewidth=2, label="Average Week")

        # Plot bounds if specified
        if plot_bounds:
            plt.fill_between(
                avg_week.index,
                min_bound[(parameter, 'min')],
                max_bound[(parameter, 'max')],
                color="gray",
                alpha=0.3,
                label="Bounds"
            )

        # Title and axis labels - Bold and bigger
        plt.title(f"{seasons[i]} - Average Week for {parameter}", fontsize=16, weight='bold')
        plt.xlabel("Hour of the Week", fontsize=14, weight='bold')
        plt.ylabel(y_label, fontsize=14, weight='bold')

        # Legend - Bold and bigger
        plt.legend(fontsize=12, loc='upper right', frameon=True)

        # Save plot with dynamic filename
        filename = f"avg_week_{seasons[i].lower()}_{parameter}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-resolution save

        # Display plot
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


import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_path = r"C:\Users\Mika\Desktop\Master\Data\Hourly_el_price_cf_wind_cf_solar_1.xlsx"

    data = load_data(file_path)
    if data is not None:
        # Unpack the returned values from create_condensed_weeks
        condensed_data, condensed_bounds = get_condensed_data(file_path)

        # Prepare data for the model
        if condensed_data is not None:
            time_varying_scenario = {
                'cf_wind': condensed_data['cf_wind'].tolist(),
                'cf_solar': condensed_data['cf_solar'].tolist(),
                'el_price': condensed_data['el_price'].tolist(),
            }



            # Pass bounds as needed for further processing or plotting
            print("Time-varying scenario and bounds prepared successfully.")
        else:
            print("Failed to generate condensed data for the model.")
