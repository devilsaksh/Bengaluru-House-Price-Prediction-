import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Streamlit UI
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# File Paths
MODEL_PATH = "real_estate_model.h5"
ENCODER_PATH = "encoder.pkl"
SCALER_PATH = "scaler.pkl"

# Load the model, encoder, and scaler with error handling
try:
    model = load_model(MODEL_PATH, compile=False)  # Load without compiling
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])  # Compile with correct settings
    st.success("Model loaded and compiled successfully!")
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found! Please check the path.")
    model = None

try:
    encoder = joblib.load(ENCODER_PATH)
    st.success("Encoder loaded successfully!")
except FileNotFoundError:
    st.error(f"Encoder file '{ENCODER_PATH}' not found! Please check the path.")
    encoder = None

try:
    scaler = joblib.load(SCALER_PATH)
    st.success("Scaler loaded successfully!")
except FileNotFoundError:
    st.error(f"Scaler file '{SCALER_PATH}' not found! Please check the path.")
    scaler = None

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2d2739;
        padding: 20px;
    }
    .stMarkdownTitle {
        font-size: 40px;
        text-align: center;
        color: white;
    }
    .stHeader {
        text-align: left;
        font-size: 28px;
        color: white;
    }
    .st-button button {
        background-color: #695b85;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='stMarkdownTitle'>HOUSE PRICE PREDICTION APP</h1>", unsafe_allow_html=True)
st.header('Input House Information')

# Streamlit UI elements for inputs
city = st.selectbox('City', ['Bangalore'])
total_sqft = st.number_input('Total Area in square feet', min_value=1)
location = st.selectbox(
    'Location',
    [ '1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar', 'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar', 'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele', 'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Badavala Nagar', 'Balagere', 'Banashankari', 'Banashankari Stage II', 'Banashankari Stage III', 'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi', 'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road', 'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur', 'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar', 'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli', 'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area', 'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar', 'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi', 'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town', 'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli', 'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi', 'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur', 'Dommasandra','EPIP Zone', 'Electronic City', 'Electronic City Phase II', 'Electronics City Phase 1', 'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar', 'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout', 'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout', 'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu', 'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu', 'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani', 'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi', 'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli', 'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala', 'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri', 'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli', 'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte', 'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate', 'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar', 'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra', 'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli', 'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya', 'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi', 'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura', 'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road', 'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur', 'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout', 'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar', 'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur', 'Sarjapur Road', 'Sarjapura - Attibele Road', 'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli', 'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya', 'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya', 'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya', 'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vasanthapura', 'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout', 'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli']  # Truncated for brevity
)
bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10)
bhk = st.number_input('BHK Value', min_value=1, max_value=10)

# Predict button
if st.button('Predict Price'):
    if model and encoder and scaler:
        try:
            # Create a DataFrame with feature names
            user_input_df = pd.DataFrame({
                'total_sqft': [total_sqft],
                'bath': [bath],
                'bhk': [bhk],
                'location': [location]  # Include the location as a separate column
            })

            # Encode the location
            user_input_df['location'] = encoder.transform(user_input_df[['location']])

            # Scale the numerical data (ensure order matches training)
            user_input_scaled = scaler.transform(user_input_df)

            # Make the prediction
            predicted_price = model.predict(user_input_scaled)[0][0]  # Extract scalar value
            
            # Display the predicted price
            st.subheader('Price Predicted:')
            st.write(f'₹ {abs(round(predicted_price, 2))} Lakhs')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Required files are not loaded. Prediction cannot be performed.")

