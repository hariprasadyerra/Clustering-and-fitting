import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)

    # Remove the 'Unnamed' column due to the trailing comma in each row
    if 'Unnamed: 67' in data.columns:
        data = data.drop(columns=['Unnamed: 67'])

    # Identify columns with missing values
    missing_value_columns = data.columns[data.isna().any()].tolist()
    print("Columns with missing values:", missing_value_columns)

    # Replace missing values in numeric columns with the mean
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[column].isna().any():
            data[column].fillna(data[column].mean(), inplace=True)

    print("Missing values filled with mean")

    # Normalization
    # Normalizing numerical features using z-score normalization
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

    print("Data normalized with z-score normalization")

    return data

def reshape_and_cluster(data, indicators, n_clusters=3, n_init=10):
    # Reshape the dataset
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = [col for col in data.columns if col not in id_vars]
    data_long = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')

    # Filter for selected indicators
    data_filtered = data_long[data_long['Indicator Name'].isin(indicators)]

    # Pivot the data to wide format
    data_pivoted = data_filtered.pivot_table(index=['Country Code', 'Year'], columns='Indicator Name', values='Value').reset_index()

    # Drop rows with any missing values in the selected indicators
    data_pivoted = data_pivoted.dropna()

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data_pivoted[indicators])

    # Add cluster information to the DataFrame
    data_pivoted['Cluster'] = clusters

    return data_pivoted

def visualize_clusters(data):
    # Ensure the Year column is treated as a numerical value for plotting
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

    # 1. Clustered Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='CO2 emissions (metric tons per capita)', y='Renewable energy consumption (% of total final energy consumption)', hue='Cluster', palette='viridis')
    plt.title('CO2 Emissions vs Renewable Energy Consumption by Cluster')
    plt.xlabel('CO2 Emissions (Metric Tons per Capita)')
    plt.ylabel('Renewable Energy Consumption (% of Total)')
    plt.show()

    # 2. Clustered Bar Chart
    plt.figure(figsize=(10, 6))
    avg_data = data.groupby('Cluster')[['Forest area (% of land area)', 'Agriculture, forestry, and fishing, value added (% of GDP)']].mean()
    avg_data.plot(kind='bar', colormap='viridis')
    plt.title('Average Forest Area and Agriculture Value by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Value (%)')
    plt.show()

    # 3. Time Series Line Plot
    plt.figure(figsize=(10, 6))
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        cluster_data.groupby('Year')['Urban population growth (annual %)'].mean().plot(label=f'Cluster {cluster}')
    plt.title('Urban Population Growth Over Time by Cluster')
    plt.xlabel('Year')
    plt.ylabel('Urban Population Growth (%)')
    plt.legend()
    plt.show()

    # 4. Cluster Heatmap
    plt.figure(figsize=(10, 6))
    cluster_avg = data.groupby('Cluster')[['CO2 emissions (metric tons per capita)', 'Urban population growth (annual %)', 'Renewable energy consumption (% of total final energy consumption)', 'Forest area (% of land area)', 'Agriculture, forestry, and fishing, value added (% of GDP)']].mean()
    sns.heatmap(cluster_avg, annot=True, cmap='coolwarm')
    plt.title('Average Indicator Values by Cluster')
    plt.xlabel('Indicator')
    plt.ylabel('Cluster')
    plt.show()

def main():
    filepath = 'dataset.csv'
    data = load_and_preprocess_data(filepath)

    print("Data loaded")
    print("===========\n")

    print("A list of Indicator Names in the dataset")
    print(data['Indicator Name'].unique())
    
    # List of indicators to use for clustering
    indicators = [
        'CO2 emissions (metric tons per capita)',
        'Urban population growth (annual %)',
        'Renewable energy consumption (% of total final energy consumption)',
        'Forest area (% of land area)',
        'Agriculture, forestry, and fishing, value added (% of GDP)'
    ]

    clustered_data = reshape_and_cluster(data, indicators)
    print("========================================\n")

    visualize_clusters(clustered_data)


if __name__ == "__main__":
    main()
