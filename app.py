import streamlit as st

# Function to label condition based on Hemoglobin, WBC, and Platelet Count and provide interactive solutions
def label_diagnosis(gender, hemoglobin, wbc, platelets):
    # 1. Check for Fatigue (Low Hemoglobin)
    if (gender == 'female' and hemoglobin < 12.0) or (gender == 'male' and hemoglobin < 13.8):
        return 'Fatigue (Low Hemoglobin)', "Are you feeling unusually tired or weak? If yes, increasing iron intake may help. Foods like spinach, lentils, and red meat are good sources of iron. You may also consider iron supplements. It's important to consult a healthcare provider for further testing."

    # 2. Check for Infection based on High WBC count
    elif wbc > 10000:
        return 'Infection (High WBC)', "Do you have symptoms like fever, chills, or body aches? If yes, this could indicate an infection. It's important to see a healthcare provider who may recommend antibiotics or other treatments based on the infection type."

    # 3. Check for Bleeding Risk (Low Platelets)
    elif platelets < 150000:
        return 'Bleeding Risk (Low Platelets)', "Are you experiencing easy bruising, frequent nosebleeds, or prolonged bleeding from small cuts? These could be signs of low platelet count. You should consult a healthcare provider to determine the cause and consider platelet transfusions or other treatments."

    # 4. Check for Bleeding Risk (High Platelets)
    elif platelets > 450000:
        return 'Bleeding Risk (High Platelets)', "Are you experiencing unusual blood clotting, swelling, or pain in your limbs? High platelets can lead to clotting issues. You should seek medical advice, and your healthcare provider might suggest blood thinners or other treatments to reduce the clotting risk."

    # Default to Normal if no conditions are met
    return 'Normal', "Your CBC values are within the normal range. Continue maintaining a balanced diet, stay active, and follow regular health check-ups."

# Streamlit UI
def main():
    st.title("CBC Report Diagnosis and Solutions")

    # User inputs for all CBC parameters
    gender = st.selectbox('Gender', ['male', 'female'])

    hemoglobin = st.number_input('Hemoglobin (g/dL)', min_value=0.0, max_value=30.0, value=13.0)
    wbc = st.number_input('White Blood Cells (cells/mcL)', min_value=0, max_value=30000, value=7000)
    platelets = st.number_input('Platelet Count (cells/mcL)', min_value=10000, max_value=1000000, value=250000)

    # Create diagnosis based on the input
    if st.button('Get Diagnosis'):
        diagnosis, solution = label_diagnosis(gender, hemoglobin, wbc, platelets)
        st.write(f"### Diagnosis: {diagnosis}")
        st.write(f"### Suggested Solution: {solution}")

        # Additional interactive questions based on diagnosis
        if diagnosis == 'Fatigue (Low Hemoglobin)':
            fatigue_symptoms = st.radio("Are you feeling unusually tired or weak?", ('Yes', 'No'))
            if fatigue_symptoms == 'Yes':
                st.write("Increasing iron intake through food like spinach, lentils, and red meat can help. Consider iron supplements after consulting your doctor.")
            else:
                st.write("Continue to maintain a healthy diet with iron-rich foods and seek medical advice if symptoms persist.")

        elif diagnosis == 'Infection (High WBC)':
            infection_symptoms = st.radio("Do you have symptoms like fever, chills, or body aches?", ('Yes', 'No'))
            if infection_symptoms == 'Yes':
                st.write("A high WBC count might indicate an infection. Visit a healthcare provider who can diagnose and suggest appropriate treatments.")
            else:
                st.write("Monitor for any other symptoms and consider following up with your doctor if you feel unwell.")

        elif diagnosis == 'Bleeding Risk (Low Platelets)':
            bleeding_symptoms = st.radio("Are you experiencing easy bruising, frequent nosebleeds, or prolonged bleeding?", ('Yes', 'No'))
            if bleeding_symptoms == 'Yes':
                st.write("Low platelets can cause bleeding issues. It's recommended to consult with a healthcare provider for further testing and potential platelet transfusions.")
            else:
                st.write("Monitor any unusual symptoms, and follow up with a healthcare provider for further guidance.")

        elif diagnosis == 'Bleeding Risk (High Platelets)':
            clotting_symptoms = st.radio("Are you experiencing unusual blood clotting, swelling, or pain in your limbs?", ('Yes', 'No'))
            if clotting_symptoms == 'Yes':
                st.write("High platelets can increase the risk of clotting. Consult a healthcare provider for evaluation, and they may suggest blood thinners or other treatments.")
            else:
                st.write("Keep an eye on any potential symptoms, and continue regular check-ups with your doctor.")

        elif diagnosis == 'Normal':
            st.write("Your CBC values are within the normal range. Keep up with a healthy lifestyle, balanced diet, and routine health check-ups.")

# Run the app
if __name__ == "__main__":
    main()
