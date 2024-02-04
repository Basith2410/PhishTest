import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pickle
import streamlit as st

# Step 1: Read the CSV files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Step 2: Combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)

# Step 3: Remove 'url' and remove duplicates, then create X and Y for the models (Supervised Learning)
df = df.drop('URL', axis=1)
df = df.drop_duplicates()
X = df.drop('label', axis=1)
Y = df['label']

# Step 4: Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Step 5: Create a Neural Network model using sklearn
nn_model = MLPClassifier(alpha=1)

# Step 6: Train the model
nn_model.fit(x_train, y_train)

# Step 7: Make some predictions using test data
predictions_nn = nn_model.predict(x_test)

# Step 8: Create a confusion matrix and tn, tp, fn , fp
tn_nn, fp_nn, fn_nn, tp_nn = confusion_matrix(y_true=y_test, y_pred=predictions_nn).ravel()

# Step 9: Calculate accuracy, precision, and recall scores
accuracy_nn = (tp_nn + tn_nn) / (tp_nn + tn_nn + fp_nn + fn_nn)
precision_nn = tp_nn / (tp_nn + fp_nn)
recall_nn = tp_nn / (tp_nn + fn_nn)

# Step 10: Save the Neural Network model
filename_nn = 'neural_network_model.sav'
pickle.dump(nn_model, open(filename_nn, 'wb'))

# Step 11: Loading the saved Neural Network model
loaded_model_nn = pickle.load(open(filename_nn, 'rb'))

# Function to check URL against datasets and predict
def check_url(url):
    # Assuming you have a function get_features_from_url(url) that extracts features.
    input_data = get_features_from_url(url)
    np_input = np.asarray(input_data)
    reshape_input = np_input.reshape(1, -1)
    prediction_nn = loaded_model_nn.predict(reshape_input)

    if prediction_nn == 1:
        return "Phishing website"
    else:
        return "Legitimate website"

# Function to get features from URL
def get_features_from_url(url):
    # Assuming you have a dataframe 'df' with features and labels, replace 'your_dataframe' with your actual dataframe.
    row = df[df['URL'] == url].iloc[0]
    features = row.drop('label').values
    return features

# Sidebar for navigation
with st.sidebar:
    url_input = st.text_input("Enter URL:")
    result = ""

    if st.button("Check URL"):
        result = check_url(url_input)

    st.write("Result:", result)
