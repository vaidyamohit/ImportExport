import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'https://raw.githubusercontent.com/vaidyamohit/ImportExport/refs/heads/main/Imports_Exports_Dataset.csv'
data = pd.read_csv(file_path)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Encode categorical variables using LabelEncoder for the predictive model
label_encoders = {}
for column in ['Category', 'Shipping_Method', 'Payment_Terms', 'Country']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target for prediction model
features = data[['Category', 'Quantity', 'Value', 'Shipping_Method', 'Payment_Terms']]
target = data['Country']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Sidebar Filters
st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect('Select Countries:', label_encoders['Country'].inverse_transform(data['Country'].unique()))
selected_categories = st.sidebar.multiselect('Select Product Categories:', label_encoders['Category'].inverse_transform(data['Category'].unique()))
selected_shipping = st.sidebar.multiselect('Select Shipping Methods:', label_encoders['Shipping_Method'].inverse_transform(data['Shipping_Method'].unique()))
selected_payment_terms = st.sidebar.multiselect('Select Payment Terms:', label_encoders['Payment_Terms'].inverse_transform(data['Payment_Terms'].unique()))
start_date = st.sidebar.date_input('Start Date', value=data['Date'].min())
end_date = st.sidebar.date_input('End Date', value=data['Date'].max())

# Filter data based on sidebar inputs
filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

if selected_countries:
    try:
        filtered_data = filtered_data[filtered_data['Country'].isin(label_encoders['Country'].transform(selected_countries))]
    except KeyError:
        st.error("Error: Selected country does not exist in the data.")

if selected_categories:
    filtered_data = filtered_data[filtered_data['Category'].isin(label_encoders['Category'].transform(selected_categories))]

if selected_shipping:
    filtered_data = filtered_data[filtered_data['Shipping_Method'].isin(label_encoders['Shipping_Method'].transform(selected_shipping))]

if selected_payment_terms:
    filtered_data = filtered_data[filtered_data['Payment_Terms'].isin(label_encoders['Payment_Terms'].transform(selected_payment_terms))]

# Main visualizations
st.title("Comprehensive Imports/Exports Dashboard")

# Box Plot
st.header("Shipment Values by Product Category (Box Plot)")
if not filtered_data.empty:
    fig_box = px.box(filtered_data, x='Category', y='Value', color='Category', title="Shipment Values by Product Category")
    st.plotly_chart(fig_box)
else:
    st.error("No data available to display the box plot.")

# Line Graph
st.header("Trends of Import/Export Values Over Time (Line Graph)")
line_data = filtered_data.groupby(['Date', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
if not line_data.empty:
    fig_line = px.line(line_data, x='Date', y='Value', color='Import_Export', title="Total Import/Export Values Over Time")
    st.plotly_chart(fig_line)
else:
    st.error("No data available to display the line graph.")

# Pie Chart
st.header("Proportion of Imports vs Exports by Country (Pie Chart)")
country_data = filtered_data.groupby(['Country', 'Import_Export']).agg({'Value': 'sum'}).reset_index()
if not country_data.empty:
    pie_data = country_data.groupby('Import_Export').agg({'Value': 'sum'}).reset_index()
    fig_pie = px.pie(pie_data, names='Import_Export', values='Value', title="Proportion of Imports vs Exports")
    st.plotly_chart(fig_pie)
else:
    st.error("No data available to display the pie chart.")

# Heatmap
st.header("Correlation Between Features (Heatmap)")
try:
    if 'Weight' in filtered_data.columns:
        correlation_data = filtered_data[['Quantity', 'Value', 'Weight']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_data, annot=True, cmap='viridis', ax=ax)
        ax.set_title('Correlation Between Features')
        st.pyplot(fig)
    else:
        st.warning("'Weight' column is missing. Unable to generate the heatmap.")
except Exception as e:
    st.error(f"An error occurred while generating the heatmap: {str(e)}")

# Bar Chart
st.header("Total Shipment Value by Shipping Method (Bar Chart)")
shipping_method_data = filtered_data.groupby('Shipping_Method').agg({'Value': 'sum'}).reset_index()
if not shipping_method_data.empty:
    fig_bar = px.bar(shipping_method_data, x='Shipping_Method', y='Value', color='Shipping_Method', title="Total Shipment Value by Shipping Method")
    st.plotly_chart(fig_bar)
else:
    st.error("No data available to display the bar chart.")

# Predictive Model Section
st.header("Predictive Model for Import/Export Decisions")

# Dropdowns for prediction
product_category = st.selectbox('Select Product Category:', options=label_encoders['Category'].inverse_transform(range(len(label_encoders['Category'].classes_))))
shipping_method = st.selectbox('Select Shipping Method:', options=label_encoders['Shipping_Method'].inverse_transform(range(len(label_encoders['Shipping_Method'].classes_))))
payment_terms = st.selectbox('Select Payment Terms:', options=label_encoders['Payment_Terms'].inverse_transform(range(len(label_encoders['Payment_Terms'].classes_))))
quantity = st.number_input('Enter Quantity:', min_value=1)
value = st.number_input('Enter Value:', min_value=1.0, step=0.01)

# Prediction
if st.button('Predict Best Country'):
    try:
        # Transform inputs using LabelEncoder
        product_category_encoded = label_encoders['Category'].transform([product_category])[0]
        shipping_method_encoded = label_encoders['Shipping_Method'].transform([shipping_method])[0]
        payment_terms_encoded = label_encoders['Payment_Terms'].transform([payment_terms])[0]

        # Prediction
        input_data = np.array([[product_category_encoded, quantity, value, shipping_method_encoded, payment_terms_encoded]])
        prediction = clf.predict(input_data)

        # Decode the predicted country
        predicted_country = label_encoders['Country'].inverse_transform(prediction)[0]

        st.success(f"The best country to import from based on your selections is: {predicted_country}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
