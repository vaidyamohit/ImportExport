import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Load the dataset
file_path = '/content/Imports_Exports_Dataset.csv'
data = pd.read_csv(file_path)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# 1. Box Plot: Distribution of shipment values by product category
def plot_box_plot(data):
    fig = px.box(data, x='Category', y='Value', title="Shipment Values by Product Category",
                 color='Category')
    fig.show()

# 2. Line Graph: Trends of total import/export values over time
def plot_line_graph(data):
    line_data = data.groupby(['Date', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    fig = px.line(line_data, x='Date', y='Value', color='Import_Export',
                  title="Total Import/Export Values Over Time")
    fig.show()

# 3. Pie Chart: Proportion of imports vs exports by country
def plot_pie_chart(data):
    country_data = data.groupby(['Country', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    pie_data = country_data.groupby('Import_Export').agg({'Value': 'sum'}).reset_index()
    fig = px.pie(pie_data, names='Import_Export', values='Value',
                 title="Proportion of Imports vs Exports")
    fig.show()

# 4. Heatmap: Correlation between features
def plot_heatmap(data):
    correlation_data = data[['Quantity', 'Value', 'Weight']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='viridis')
    plt.title('Correlation Between Features')
    plt.show()

# 5. Bar Chart: Total shipment value by shipping method
def plot_bar_chart(data):
    shipping_method_data = data.groupby('Shipping_Method').agg({'Value': 'sum'}).reset_index()
    fig = px.bar(shipping_method_data, x='Shipping_Method', y='Value',
                 title="Total Shipment Value by Shipping Method",
                 color='Shipping_Method')
    fig.show()

# Run all the plots
plot_box_plot(data)
plot_line_graph(data)
plot_pie_chart(data)
plot_heatmap(data)
plot_bar_chart(data)
