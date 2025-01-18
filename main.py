import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('health_issue_model.pkl')

# List of symptoms
symptom_list = [
    "Sudden Fever", "Headache", "Mouth Bleed", "Nose Bleed", "Muscle Pain", "Joint Pain",
    "Vomiting", "Rash", "Diarrhea", "Hypotension", "Pleural Effusion", "Ascites", "Gastro Bleeding",
    "Swelling", "Nausea", "Chills", "Myalgia", "Digestion Trouble", "Fatigue", "Skin Lesions",
    "Stomach Pain", "Orbital Pain", "Neck Pain", "Weakness", "Back Pain", "Weight Loss",
    "Gum Bleed", "Jaundice", "Coma", "Dizziness", "Inflammation", "Red Eyes", "Loss of Appetite",
    "Urination Loss", "Slow Heart Rate", "Abdominal Pain", "Light Sensitivity", "Yellow Skin",
    "Yellow Eyes", "Facial Distortion", "Microcephaly", "Rigor", "Bitter Tongue", "Convulsion",
    "Anemia", "Cocacola Urine", "Hypoglycemia", "Prostration", "Hyperpyrexia", "Stiff Neck",
    "Irritability", "Confusion", "Tremor", "Paralysis", "Lymph Swells", "Breathing Restriction",
    "Toe Inflammation", "Finger Inflammation", "Lips Irritation", "Itchiness", "Ulcers", "Toenail Loss",
    "Speech Problem", "Bullseye Rash"
]

# Streamlit app title
st.title('Health Issue Prediction Based on Symptoms')

# Custom CSS styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to display prediction explanation and health tips
def show_explanation(prediction):
    if prediction == "Dengue":
        st.write("### Explanation:")
        st.write("Dengue fever is caused by a virus transmitted through mosquito bites. It typically involves high fever and severe joint pain.")
        st.write("### Health Tips:")
        st.write("Stay hydrated, avoid mosquito bites, and consult a healthcare provider for further tests.")
    elif prediction == "Malaria":
        st.write("### Explanation:")
        st.write("Malaria is caused by a parasite spread by mosquitoes. Common symptoms include fever, chills, and sweating.")
        st.write("### Health Tips:")
        st.write("Seek medical attention for diagnosis and appropriate treatment, typically antimalarial drugs.")
    elif prediction == "Zika":
        st.write("### Explanation:")
        st.write("Zika virus is transmitted through mosquitoes and can cause fever, rash, and joint pain.")
        st.write("### Health Tips:")
        st.write("Stay hydrated and rest. Avoid mosquito bites to prevent further infection.")
    else:
        st.write("### Explanation:")
        st.write(f"The prediction '{prediction}' requires further medical investigation. Please consult a healthcare provider.")
        st.write("### Health Tips:")
        st.write("It is important to consult a doctor for accurate diagnosis and treatment.")

# Sidebar for symptom selection
with st.sidebar:
    st.header('Symptom Selection')
    selected_symptoms = [st.checkbox(symptom) for symptom in symptom_list]

# Button to predict health issue
if st.button("Predict Health Issue"):
    # Check if at least one symptom is selected
    if not any(selected_symptoms):
        st.error("Please select at least one symptom before predicting.")
    else:
        # Convert symptoms input into a numpy array (1 for symptom present, 0 for absent)
        symptoms = np.array(selected_symptoms)

        # Ensure symptoms length is correct
        if symptoms.shape[0] == 64:
            # Show loading spinner while making prediction
            with st.spinner('Making prediction...'):
                prediction = model.predict(symptoms.reshape(1, -1))

            # Show prediction result
            st.success(f"Predicted Health Issue: **{prediction[0]}**")
            
            # Display the explanation and health tips for the prediction
            show_explanation(prediction[0])
            
            # Visualize the prediction probabilities
            prediction_probs = model.predict_proba(symptoms.reshape(1, -1))[0]
            labels = model.classes_

            fig, ax = plt.subplots()
            ax.barh(labels, prediction_probs)
            ax.set_xlabel('Probability')
            ax.set_title('Health Issue Prediction Probabilities')

            st.pyplot(fig)
            
        else:
            st.error("There was an issue with the input data. Please check the symptoms again.")
