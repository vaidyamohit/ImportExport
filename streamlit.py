import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'https://raw.githubusercontent.com/vaidyamohit/ImportExport/refs/heads/main/Imports_Exports_Dataset.csv'
data = pd.read_csv(file_path)

# Check if dataset is loaded correctly
if data.empty:
    st.error("Dataset could not be loaded. Please check the file path or dataset.")
else:
    # Convert the Date column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_countries = st.sidebar.multiselect('Select Countries:', data['Country'].unique())
    selected_categories = st.sidebar.multiselect('Select Product Categories:', data['Category'].unique())
    selected_shipping = st.sidebar.multiselect('Select Shipping Methods:', data['Shipping_Method'].unique())
    selected_payment_terms = st.sidebar.multiselect('Select Payment Terms:', data['Payment_Terms'].unique())
    start_date = st.sidebar.date_input('Start Date', value=data['Date'].min())
    end_date = st.sidebar.date_input('End Date', value=data['Date'].max())

    # Filter data based on sidebar inputs
    filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

    if selected_countries:
        filtered_data = filtered_data[filtered_data['Country'].isin(selected_countries)]

    if selected_categories:
        filtered_data = filtered_data[filtered_data['Category'].isin(selected_categories)]

    if selected_shipping:
        filtered_data = filtered_data[filtered_data['Shipping_Method'].isin(selected_shipping)]

    if selected_payment_terms:
        filtered_data = filtered_data[filtered_data['Payment_Terms'].isin(selected_payment_terms)]

    # Main visualizations
    st.title("Comprehensive Imports/Exports Dashboard")

    # Box Plot
    st.header("Shipment Values by Product Category (Box Plot)")
    if not filtered_data.empty:
        fig_box = px.box(filtered_data, x='Category', y='Value', color='Category', title="Shipment Values by Product Category")
        st.plotly_chart(fig_box)
    else:
        st.error("No data available for the selected filters to display the box plot.")

    # Line Graph
    st.header("Trends of Import/Export Values Over Time (Line Graph)")
    line_data = filtered_data.groupby(['Date', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    if not line_data.empty:
        fig_line = px.line(line_data, x='Date', y='Value', color='Import_Export', title="Total Import/Export Values Over Time")
        st.plotly_chart(fig_line)
    else:
        st.error("No data available for the selected filters to display the line graph.")

    # Pie Chart
    st.header("Proportion of Imports vs Exports by Country (Pie Chart)")
    country_data = filtered_data.groupby(['Country', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
    if not country_data.empty:
        pie_data = country_data.groupby('Import_Export').agg({'Value': 'sum'}).reset_index()
        fig_pie = px.pie(pie_data, names='Import_Export', values='Value', title="Proportion of Imports vs Exports")
        st.plotly_chart(fig_pie)
    else:
        st.error("No data available for the selected filters to display the pie chart.")

    # Heatmap
    st.header("Correlation Between Features (Heatmap)")
    if 'Weight' in filtered_data.columns:
        correlation_data = filtered_data[['Quantity', 'Value', 'Weight']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_data, annot=True, cmap='viridis', ax=ax)
        ax.set_title('Correlation Between Features')
        st.pyplot(fig)
    else:
        st.warning("The 'Weight' column is missing from the dataset. Unable to generate the heatmap.")

    # Bar Chart
    st.header("Total Shipment Value by Shipping Method (Bar Chart)")
    shipping_method_data = filtered_data.groupby('Shipping_Method').agg({'Value': 'sum'}).reset_index()
    if not shipping_method_data.empty:
        fig_bar = px.bar(shipping_method_data, x='Shipping_Method', y='Value', color='Shipping_Method', title="Total Shipment Value by Shipping Method")
        st.plotly_chart(fig_bar)
    else:
        st.error("No data available for the selected filters to display the bar chart.")

    # ------------------- PREDICTIVE MODEL SECTION -------------------

    st.header("Predictive Model for Import/Export Shipment Value")

    # Define the regression formula
    # We'll predict the 'Value' of shipment based on several features
    lin_reg_model = smf.ols('Value ~ Quantity + Weight + Import_Export + Shipping_Method + Payment_Terms', data=data).fit()

    # Input widgets for user input with float values
    quantity = st.number_input("Enter Quantity:", min_value=1, step=1)
    weight = st.number_input("Enter Weight:", min_value=0.01, step=0.01)
    import_export = st.selectbox("Import or Export", options=data['Import_Export'].unique())
    shipping_method = st.selectbox("Select Shipping Method", options=data['Shipping_Method'].unique())
    payment_terms = st.selectbox("Select Payment Terms", options=data['Payment_Terms'].unique())

    # Button to trigger prediction
    predict_button = st.button("Predict Shipment Value")

    # Output section for prediction result
    if predict_button:
        input_data = pd.DataFrame({
            'Quantity': [quantity],
            'Weight': [weight],
            'Import_Export': [import_export],
            'Shipping_Method': [shipping_method],
            'Payment_Terms': [payment_terms]
        })

        # Predict using the linear regression model
        prediction = lin_reg_model.predict(input_data)

        st.success(f'Predicted Shipment Value: {prediction.iloc[0]:.2f}')
