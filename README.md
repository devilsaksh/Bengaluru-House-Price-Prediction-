# Bengaluru-House-Price-Prediction-

# Bengaluru-House-Price-Prediction

# Real Estate Price Prediction

A machine learning-based web application for predicting real estate prices using features such as total square footage, number of bathrooms, number of bedrooms, and location. The application uses a neural network model built with TensorFlow to predict the price of a property.

## Features
- **Data Preprocessing**: The dataset includes features like total square footage, number of bathrooms, number of bedrooms (BHK), and location.
- **Model**: A neural network model built using TensorFlow for regression.
- **Scaling & Encoding**: Scales numeric features and encodes categorical location data.
- **Prediction**: Uses a trained model to predict property prices based on input features.

## Project Structure

- **app/streamlit_app.py**: Main code for the interactive Streamlit web application.
- **models/**: Contains the trained Keras model (`real_estate_model.h5`), scaler object (`scaler.pkl`), and encoder object (`encoder.pkl`).
- **data/Cleaned_data.csv**: The dataset used for training the model.
- **requirements.txt**: Python dependencies required for the project.
- **README.md**: Documentation for the project.
- **.gitignore**: Files and directories to exclude from version control.

## Dependencies
Make sure to install all the necessary dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt

## To run the project 
streamlit run app.py 
