import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config

# Load and scale the data as before
data_file_path = r"C:\Users\Mika\Desktop\Master\Data/Hourly_el_price_cf_wind_cf_solar.xlsx"
data = pd.read_excel(data_file_path)

if config.sampling_method == 'block_bootstrapping':
    # Define block bootstrapping function
    def block_bootstrap(data, block_size, n_samples):
        """Perform block bootstrapping on the time series data."""
        n_blocks = int(np.ceil(len(data) / block_size))
        bootstrapped_data = []

        for _ in range(n_samples):
            blocks = []
            for _ in range(n_blocks):
                # Randomly select a start index for the block
                start_idx = np.random.randint(0, len(data) - block_size + 1)
                block = data[start_idx:start_idx + block_size]
                blocks.append(block)
            # Concatenate blocks to form the bootstrapped sample
            bootstrapped_sample = np.concatenate(blocks, axis=0)[:len(data)]
            bootstrapped_data.append(bootstrapped_sample)

        return np.array(bootstrapped_data)


    # Function to get the bootstrapped dynamic parameter data
    def get_bootstrapped_data(block_size, n_samples):
        """Returns bootstrapped time-varying parameter samples for each dynamic parameter."""
        el_price_data = data['el_price'].values
        cf_wind_data = data['cf_wind'].values
        cf_solar_data = data['cf_solar'].values

        # Perform block bootstrapping for each dynamic parameter
        el_price_bootstrapped = block_bootstrap(el_price_data, block_size, n_samples)
        cf_wind_bootstrapped = block_bootstrap(cf_wind_data, block_size, n_samples)
        cf_solar_bootstrapped = block_bootstrap(cf_solar_data, block_size, n_samples)

        return {
            "el_price": el_price_bootstrapped,
            "cf_wind": cf_wind_bootstrapped,
            "cf_solar": cf_solar_bootstrapped
        }


    # Example usage of block bootstrapping function
    def plot_bootstrapped_samples(samples, title):
        """Plot a few bootstrapped samples for visualization."""
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(samples[:5]):  # Plot only the first 5 samples
            plt.plot(sample, label=f'Sample {i + 1}')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()




    bootstrapped_data = get_bootstrapped_data(config.block_size, config.n_samples_bs)
    plot_bootstrapped_samples(bootstrapped_data['el_price'], 'Bootstrapped Electricity Prices')
    plot_bootstrapped_samples(bootstrapped_data['cf_wind'], 'Bootstrapped Wind Capacity Factors')
    plot_bootstrapped_samples(bootstrapped_data['cf_solar'], 'Bootstrapped Solar Capacity Factors')

elif config.sampling_method == 'clustering':
    # Standardize the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['el_price', 'cf_wind', 'cf_solar']])

    # Apply clustering
    n_clusters = config.n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_state)

    clusters = kmeans.fit_predict(scaled_data)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = clusters

    # Collect the data points for each cluster
    cluster_data_dict = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(clusters):
        cluster_data_dict[label].append(scaled_data[idx])


    # Plot the clusters (optional)
    def plot_clusters(scaled_data, clusters):
        plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
        plt.title("K-Means Clustering for Time-Varying Parameters")
        plt.xlabel("Electricity Price (scaled)")
        plt.ylabel("Capacity Factors (scaled)")
        plt.show()


    def sample_from_cluster_random(cluster, param, scaler):
        """Sample a random data point from a randomly chosen cluster and revert to the original scale."""
        # Select a random data point from the cluster
        random_index = np.random.randint(len(cluster))
        sample_scaled = cluster[random_index]

        # Reconvert the selected sample to the original scale using the scaler
        sample_original = scaler.inverse_transform([sample_scaled])[0]

        # Return the value corresponding to the parameter
        if param == "el_price":
            return sample_original[0]  # Electricity price in original scale
        elif param == "cf_wind":
            return sample_original[1]  # Wind capacity factor in original scale
        elif param == "cf_solar":
            return sample_original[2]  # Solar capacity factor in original scale


    def sample_from_cluster_structured(cluster, selection_value, param, scaler):
        """Sample from a cluster with structured sampling bias towards lower or upper percentiles."""
        # Sort the data points based on the selected parameter index
        if param == "el_price":
            index = 0
        elif param == "cf_wind":
            index = 1
        elif param == "cf_solar":
            index = 2

        sorted_cluster = sorted(cluster, key=lambda x: x[index])

        # Determine the selection point based on the structured sampling value (0-1)
        selection_index = int(selection_value * len(sorted_cluster))
        sample_scaled = sorted_cluster[selection_index]

        # Reconvert the selected sample to the original scale using the scaler
        sample_original = scaler.inverse_transform([sample_scaled])[0]

        # Return the value corresponding to the parameter
        return sample_original[index]


    def get_clustered_data():
        cluster_data_dict = {}
        for i in range(n_clusters):
            cluster_data_dict[i] = data[clusters == i]

        return {
            "el_price": {"data": [cluster_data_dict[i]['el_price'].values for i in range(n_clusters)]},
            "cf_wind": {"data": [cluster_data_dict[i]['cf_wind'].values for i in range(n_clusters)]},
            "cf_solar": {"data": [cluster_data_dict[i]['cf_solar'].values for i in range(n_clusters)]},
        }, scaler


    def reconvert_to_original(param, standardized_value, scaler):
        """Reconvert a standardized value back to its original scale."""
        if param == "el_price":
            column_index = 0
        elif param == "cf_wind":
            column_index = 1
        elif param == "cf_solar":
            column_index = 2
        else:
            raise ValueError("Unknown parameter")

        # Prepare a dummy array for inverse transformation
        dummy_array = np.zeros((1, 3))
        dummy_array[0, column_index] = standardized_value

        # Apply inverse transformation
        original_value = scaler.inverse_transform(dummy_array)[0, column_index]
        return original_value


    # Plot function call (optional)
    plot_clusters(scaled_data, clusters)

elif config.sampling_method == 'simple_slice':
    # Define the simple slice function
    def simple_slice(data, length, n_samples):
        """Take random slices from the dataset with the specified length."""
        sliced_data = []

        for _ in range(n_samples):
            start_idx = np.random.randint(0, len(data) - length + 1)
            slice_sample = data[start_idx:start_idx + length]
            sliced_data.append(slice_sample)

        return np.array(sliced_data)

        # Simple slicing


    def get_simple_slice_data(time_period_length):
        """Generates a random slice from the time series data for each time-varying parameter."""
        start_idx = np.random.randint(0, len(data) - time_period_length + 1)
        slice_sample = {
            "el_price": data['el_price'].values[start_idx:start_idx + time_period_length],
            "cf_wind": data['cf_wind'].values[start_idx:start_idx + time_period_length],
            "cf_solar": data['cf_solar'].values[start_idx:start_idx + time_period_length]
        }
        return slice_sample




