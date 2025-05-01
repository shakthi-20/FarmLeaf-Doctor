import streamlit as st
from PIL import Image
import datetime
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from io import BytesIO
import os
import serial
import re
import time

# Set page config
st.set_page_config(
    page_title="FarmLeaf Doctor - Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load your trained CNN model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_plant_model():
    try:
        model = load_model('plant_doctor.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_plant_model()
if model is not None:
    st.success("Model loaded successfully!")

# Environmental factor ranges for different diseases
ENVIRONMENTAL_FACTORS = {
    "bacterial": {
        "temp_range": (32, 36),
        "humidity_range": (75, 100),
        "soil_moisture_range": (80, 85)
    },
    "fungus": {
        "temp_range": (25, 36),
        "humidity_range": (60, 89),
        "soil_moisture_range": (90, 100)
    },
    "virus": {
        "temp_range": (24, 30),
        "humidity_range": (50, 80),
        "soil_moisture_range": (60, 60)
    },
    "pest": {
        "temp_range": (22, 32),
        "humidity_range": (40, 70),
        "soil_moisture_range": (20, 30)
    }
}

# Check if environmental factors contribute to disease
def check_environmental_factors(disease, temp, humidity, soil_moisture):
    if disease not in ENVIRONMENTAL_FACTORS or disease == "healthy":
        return []
    
    factors = ENVIRONMENTAL_FACTORS[disease]
    contributing_factors = []
    
    if not (factors["temp_range"][0] <= temp <= factors["temp_range"][1]):
        contributing_factors.append(f"Temperature ({temp}°C) is outside ideal range ({factors['temp_range'][0]}°C-{factors['temp_range'][1]}°C)")
    
    if not (factors["humidity_range"][0] <= humidity <= factors["humidity_range"][1]):
        contributing_factors.append(f"Humidity ({humidity}%) is outside ideal range ({factors['humidity_range'][0]}%-{factors['humidity_range'][1]}%)")
    
    if not (factors["soil_moisture_range"][0] <= soil_moisture <= factors["soil_moisture_range"][1]):
        contributing_factors.append(f"Soil moisture ({soil_moisture}%) is outside ideal range ({factors['soil_moisture_range'][0]}%-{factors['soil_moisture_range'][1]}%)")
    
    return contributing_factors

# Prediction function using your model
def predict_image(image):
    """Predict disease from image using your CNN model"""
    if model is None:
        return "error", 0.0
    
    try:
        # Convert to RGB if not already (handles grayscale, RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize to model's expected input
        img = image.resize((150, 150))
        img_array = np.array(img)
        
        # Normalize and add batch dimension
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Now (1, 150, 150, 3)
        
        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        
        classes = ['bacterial', 'fungus', 'healthy', 'pest', 'virus']
        disease = classes[class_index]
        
        return disease, float(confidence)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "error", 0.0

# Database of plant disease remedies
remedies_db = {
    "bacterial": {
        "display_name": "Bacterial Infection (Class: Bacterial)",
        "treatment": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based bactericides",
            "Use streptomycin spray for severe cases",
            "Avoid overhead watering"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Practice 3-year crop rotation",
            "Disinfect tools with bleach solution",
            "Improve soil drainage"
        ],
        "resources": ["https://www.apsnet.org"],
        "severity": "High",
        "affected_plants": ["Tomatoes", "Peppers", "Cucurbits"],
        "symptoms": [
            "Water-soaked lesions",
            "Yellow halos around spots",
            "Oozing bacterial slime",
            "Wilting of shoots"
        ]
    },
    "fungus": {
        "display_name": "Fungal Infection (Class: Fungus)",
        "treatment": [
            "Apply fungicides containing chlorothalonil or copper",
            "Remove severely infected plants immediately",
            "Apply baking soda solution (1 tbsp/gallon water) for mild cases",
            "Use sulfur dust for organic control"
        ],
        "prevention": [
            "Water early in the day to allow leaves to dry",
            "Maintain proper plant spacing (30-45cm between plants)",
            "Apply mulch to prevent soil splash",
            "Remove plant debris after harvest"
        ],
        "resources": ["https://www.apsnet.org/edcenter/disandpath/fungalasco/topics/Pages/Overview.aspx"],
        "severity": "Moderate-High",
        "affected_plants": ["Most crops", "Ornamentals", "Turf grass"],
        "symptoms": [
            "Powdery white or gray growth on leaves",
            "Circular spots with concentric rings",
            "Yellowing leaves that drop prematurely",
            "Fuzzy mold on undersides of leaves"
        ]
    },
    "healthy": {
        "display_name": "Healthy Plant (Class: Healthy)",
        "treatment": ["No treatment needed - maintain current care"],
        "prevention": [
            "Continue regular monitoring (weekly checks recommended)",
            "Maintain balanced fertilization (N-P-K ratio 10-10-10)",
            "Practice good sanitation (clean tools after use)",
            "Rotate crops preventatively (change plant families each season)"
        ],
        "resources": [],
        "severity": "None",
        "affected_plants": ["All plants"],
        "symptoms": [
            "Vibrant green coloration",
            "Firm stems and leaves",
            "Steady new growth (2-4 inches per week for most crops)",
            "No visible lesions or spots"
        ]
    },
    "pest": {
        "display_name": "Pest Damage (Class: Pest)",
        "treatment": [
            "Identify specific pest first (use magnifying lens)",
            "Apply neem oil (2 tbsp/gallon) or insecticidal soap",
            "Use yellow sticky traps for flying insects (place every 10 sq ft)",
            "Introduce beneficial insects like ladybugs (1,000 per acre)"
        ],
        "prevention": [
            "Use floating row covers (Agribon-15 recommended)",
            "Encourage predator habitats (bird houses, beneficial plants)",
            "Rotate pesticides to prevent resistance (change active ingredients)",
            "Remove weeds that harbor pests (especially nightshades)"
        ],
        "resources": ["https://extension.umn.edu/yard-and-garden-insects"],
        "severity": "Variable",
        "affected_plants": ["Vegetables", "Fruits", "Ornamentals"],
        "symptoms": [
            "Chewed leaf edges or holes (irregular patterns)",
            "Stippling or silvering of leaves (thrips damage)",
            "Sticky honeydew residue (aphid/scale indicator)",
            "Visible insects or eggs on undersides (inspect daily)"
        ]
    },
    "virus": {
        "display_name": "Viral Infection (Class: Virus)",
        "treatment": [
            "Remove and burn infected plants immediately",
            "Control aphids and whiteflies (use pyrethrin sprays)",
            "Disinfect tools with 70% alcohol between plants",
            "No chemical cure available - focus on prevention"
        ],
        "prevention": [
            "Plant resistant varieties when available (look for VFNT notations)",
            "Use insect netting (50 mesh) to block vectors",
            "Remove weed hosts within 100m of fields",
            "Purchase certified virus-free plants (check certification)"
        ],
        "resources": ["https://www.apsnet.org/edcenter/disandpath/viral/topics/Pages/PlantViruses.aspx"],
        "severity": "Very High",
        "affected_plants": ["Tomatoes", "Cucumbers", "Tobacco", "Potatoes"],
        "symptoms": [
            "Mosaic or mottled leaf patterns (light/dark green)",
            "Severe stunting and distortion (crinkled leaves)",
            "Yellow vein banding (follows leaf veins)",
            "Necrotic ring spots (dead tissue circles)"
        ]
    }
}

# Function to read sensor data from serial port
def get_sensor_data(ser):
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        while "<start>" not in line or "<end>" not in line:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
        match = re.search(r"<start>(.*?)<end>", line)
        if match:
            return match.group(1)
    except Exception as e:
        st.error(f"Sensor reading error: {str(e)}")
        return None
    return None

# Main app layout
def main():
    st.title("🌱 FarmLeaf Doctor - Plant Disease Detection")
    st.markdown("""
    <style>
    .severity-high { color: #d9534f; font-weight: bold; }
    .severity-moderate { color: #f0ad4e; font-weight: bold; }
    .severity-low { color: #5bc0de; font-weight: bold; }
    .severity-none { color: #5cb85c; font-weight: bold; }
    .diagnosis-card {
        display: flex;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .diagnosis-image { flex: 1; max-width: 150px; }
    .diagnosis-details { flex: 2; }
    .emergency-card {
        background-color: #fff8f8;
        border-left: 4px solid #ff4b4b;
        padding: 1rem;
        margin: 1rem 0;
    }
    .environment-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sensor-data {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize serial connection
    try:
        ser = serial.Serial('COM6', 115200, timeout=1)  # Change COM port as needed
        time.sleep(2)  # Wait for connection to establish
        sensor_connected = True
    except Exception as e:
        st.warning(f"⚠️ Could not connect to Arduino: {str(e)}. Using default values.")
        sensor_connected = False
        default_temp = 25
        default_humidity = 50
        default_soil_moisture = 50

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Leaf Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

            # Environmental factors from sensors
            st.subheader("🌡️ Current Environmental Conditions")
            
            if sensor_connected:
                try:
                    data = get_sensor_data(ser)
                    if data:
                        parts = dict(x.split("=") for x in data.split(","))
                        current_temp = float(parts.get('temperature', 0))
                        current_humidity = float(parts.get('humidity', 0))
                        current_soil_moisture = int(parts.get('soil_moisture', 0))
                        
                        with st.container():
                            st.markdown('<div class="sensor-data">', unsafe_allow_html=True)
                            st.metric("🌡️ Temperature", f"{current_temp:.1f} °C")
                            st.metric("💧 Humidity", f"{current_humidity:.1f} %")
                            st.metric("🌱 Soil Moisture", f"{current_soil_moisture} %")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Could not read sensor data. Using default values.")
                        current_temp = default_temp
                        current_humidity = default_humidity
                        current_soil_moisture = default_soil_moisture
                except Exception as e:
                    st.error(f"Sensor error: {str(e)}")
                    current_temp = default_temp
                    current_humidity = default_humidity
                    current_soil_moisture = default_soil_moisture
            else:
                current_temp = default_temp
                current_humidity = default_humidity
                current_soil_moisture = default_soil_moisture
            
            if st.button("🔍 Diagnose Disease"):
                if model is None:
                    st.error("Model failed to load. Please check plant_doctor.h5 exists.")
                else:
                    with st.spinner("Analyzing leaf..."):
                        disease, confidence = predict_image(image)
                        if disease == "error":
                            st.error("Analysis failed. Try another image.")
                        else:
                            st.session_state.diagnosis = {
                                "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "disease": disease,
                                "confidence": confidence,
                                "image": image,
                                "environment": {
                                    "temp": current_temp,
                                    "humidity": current_humidity,
                                    "soil_moisture": current_soil_moisture
                                }
                            }

    with col2:
        st.header("Diagnosis Results")
        
        if "diagnosis" not in st.session_state:
            st.info("Upload a leaf image and click 'Diagnose Disease'")
        else:
            diagnosis = st.session_state.diagnosis
            disease_info = remedies_db[diagnosis["disease"]]
            
            # Diagnosis card
            st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
            st.markdown('<div class="diagnosis-image">', unsafe_allow_html=True)
            st.image(diagnosis["image"], width=150)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="diagnosis-details">', unsafe_allow_html=True)
            st.markdown(f"**Condition:** {disease_info['display_name']}")
            st.markdown(f"**Confidence:** {diagnosis['confidence']*100:.1f}%")
            st.markdown(f"**Severity:** <span class='severity-{disease_info['severity'].split()[0].lower()}'>"
                    f"{disease_info['severity']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Date:** {diagnosis['datetime']}")
            st.markdown('</div></div>', unsafe_allow_html=True)

            # Display current environmental conditions - REPLACED NESTED COLUMNS
            st.markdown("### 🌍 Current Environmental Conditions")
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; flex: 1; margin: 0 5px;">
                    <div style="font-size: 0.8em; color: #000;">Temperature</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{diagnosis['environment']['temp']} °C</div>
                </div>
                <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; flex: 1; margin: 0 5px;">
                    <div style="font-size: 0.8em; color: #000;">Humidity</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{diagnosis['environment']['humidity']} %</div>
                </div>
                <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; flex: 1; margin: 0 5px;">
                    <div style="font-size: 0.8em; color: #000;">Soil Moisture</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{diagnosis['environment']['soil_moisture']} %</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
       

            # Symptoms
            st.markdown("### 🔍 Common Symptoms")
            for symptom in disease_info["symptoms"]:
                st.markdown(f"- {symptom}")
            
            # Treatment and Prevention
            tab1, tab2 = st.tabs(["🛠️ Treatment", "🛡️ Prevention"])
            with tab1:
                for treatment in disease_info["treatment"]:
                    st.markdown(f"- {treatment}")
            with tab2:
                for prevention in disease_info["prevention"]:
                    st.markdown(f"- {prevention}")
                
                # Add environmental-specific prevention if disease is detected
                if diagnosis["disease"] != "healthy":
                    env_factors = check_environmental_factors(
                        diagnosis["disease"],
                        diagnosis["environment"]["temp"],
                        diagnosis["environment"]["humidity"],
                        diagnosis["environment"]["soil_moisture"]
                    )
                    
                    if env_factors:
                        st.markdown("### 🌍 Environmental Adjustments")
                        if "Temperature" in " ".join(env_factors):
                            st.markdown("- Adjust shading or ventilation to maintain optimal temperature")
                        if "Humidity" in " ".join(env_factors):
                            st.markdown("- Improve air circulation with fans or spacing to reduce humidity")
                        if "Soil moisture" in " ".join(env_factors):
                            st.markdown("- Adjust irrigation schedule to maintain optimal soil moisture")

            if disease_info["resources"]:
                st.markdown("### 📚 Expert Resources")
                st.markdown(f"[Visit Expert Portal]({disease_info['resources'][0]})")

    # Sidebar
    with st.sidebar:
        st.header("Farmer Assistance")
        
        st.markdown("""
        <div class="emergency-card">
        <h3>📞 Emergency Plant Helpline</h3>
        <p><a href="tel:+918610318574">+91 8610318574</a></p>
        <p>24/7 support</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3>🛒 Nearest Agri-Store</h3>
        <p><strong>Amrita University Agricultural Center</strong></p>
        <p>Amritanagar, Ettimadai</p>
        <p>Tamil Nadu 641112</p>
        <p><a href="https://maps.google.com/?q=Amrita+University" target="_blank">Get Directions</a></p>
        <p><a href="tel:+918610318574">📱 Call Store</a></p>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        ### 🌦️ Current Disease Risk
        **High Risk:** Fungal diseases  
        **Moderate Risk:** Bacterial spots  
        **Updated: """ + datetime.datetime.now().strftime("%d %b %Y") + """</p>
        """, unsafe_allow_html=True)

    # Close serial connection when done
    if 'ser' in locals() and sensor_connected:
        ser.close()

if __name__ == "__main__":
    main() 