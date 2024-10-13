import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'https://raw.githubusercontent.com/vaidyamohit/ImportExport/refs/heads/main/Imports_Exports_Dataset.csv'
data = pd.read_csv(file_path)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# 1. Box Plot: Distribution of shipment values by product category
def plot_box_plot(data):
    fig = px.box(data, x='Category', y='Value', title="Shipment Values by Product Category",
                 color='Category')
    st.plotly_chart(fig)

# 2. Line Graph: Trends of total import/export values over time
def plot_line_graph(data):
    line_data = data.groupby(['Date', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    fig = px.line(line_data, x='Date', y='Value', color='Import_Export',
                  title="Total Import/Export Values Over Time")
    st.plotly_chart(fig)

# 3. Pie Chart: Proportion of imports vs exports by country
def plot_pie_chart(data):
    country_data = data.groupby(['Country', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    pie_data = country_data.groupby('Import_Export').agg({'Value': 'sum'}).reset_index()
    fig = px.pie(pie_data, names='Import_Export', values='Value',
                 title="Proportion of Imports vs Exports")
    st.plotly_chart(fig)

# 4. Heatmap: Correlation between features
def plot_heatmap(data):
    correlation_data = data[['Quantity', 'Value', 'Weight']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_data, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Correlation Between Features')
    st.pyplot(fig)

# 5. Bar Chart: Total shipment value by shipping method
def plot_bar_chart(data):
    shipping_method_data = data.groupby('Shipping_Method').agg({'Value': 'sum'}).reset_index()
    fig = px.bar(shipping_method_data, x='Shipping_Method', y='Value',
                 title="Total Shipment Value by Shipping Method",
                 color='Shipping_Method')
    st.plotly_chart(fig)

# Streamlit App Layout
st.title("Comprehensive Imports/Exports Visualizations")

# Running the visualizations
st.header("Shipment Values by Product Category (Box Plot)")
plot_box_plot(data)

st.header("Trends of Import/Export Values Over Time (Line Graph)")
plot_line_graph(data)

st.header("Proportion of Imports vs Exports (Pie Chart)")
plot_pie_chart(data)

st.header("Correlation Between Features (Heatmap)")
plot_heatmap(data)

st.header("Total Shipment Value by Shipping Method (Bar Chart)")
plot_bar_chart(data)
