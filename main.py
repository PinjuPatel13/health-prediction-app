import streamlit as st
import re
import json
import numpy as np
import cv2
from PIL import Image
from paddleocr import PaddleOCR
import plotly.express as px
import pandas as pd
from pdf2image import convert_from_bytes
from transformers import pipeline
import os

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load pre-trained BioBERT model for medical text classification
#medical_ai = pipeline("text-classification", model="distilbert-base-uncased")
medical_ai = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# Load past medical reports (if available)
db_file = "diagnosed.csv"
if os.path.exists(db_file):
    past_reports = pd.read_csv(db_file)
else:
    past_reports = pd.DataFrame(columns=["report_text", "predicted_disease"])


def advanced_preprocess_image(image):
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Apply contrast adjustment
    alpha = 1.5  # Increase contrast
    beta = 10    # Increase brightness
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(adjusted, None, 30, 7, 21)

    # Apply adaptive thresholding for better text extraction
    processed = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return processed


def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return threshold

def save_ocr_results_to_json(result, json_file="extracted_data.json"):
    extracted_data = []
    for line in result[0]:  
        text = line[1][0].strip()  
        extracted_data.append(text)
    json_data = {"extracted_text": extracted_data}
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4)
    return json_file

def extract_values_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    extracted_text = data.get("extracted_text", [])
    extracted_text = st.text_area("Edit extracted text if needed:", '\n'.join(extracted_text)).split('\n')
    
    parameters = {}
    
    test_mapping = {
        "wbc": ["wbc", "total wbc count", "white blood cells"],
        "lymp": ["lymp", "lymphocytes percentage"],
        "neutp": ["neutp", "neutrophils percentage"],
        "lymn": ["lymn", "lymphocytes count"],
        "neutn": ["neutn", "neutrophils count"],
        "rbc": ["rbc", "red blood cells"],
        "hgb": ["hgb", "hemoglobin"],
        "hct": ["hct", "hematocrit"],
        "mcv": ["mcv", "mean corpuscular volume"],
        "mch": ["mch", "mean corpuscular hemoglobin"],
        "mchc": ["mchc", "mean corpuscular hemoglobin concentration"],
        "plt": ["plt", "platelets", "platelet count"],
        "pdw": ["pdw", "platelet distribution width"],
        "pct": ["pct", "procalcitonin"],
    }

    for i in range(len(extracted_text) - 1):
        key = extracted_text[i].strip().lower()
        value = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", extracted_text[i + 1].strip())
        value = float(value[0]) if value else None
        
        for standard_name, variations in test_mapping.items():
            if key in variations or key == standard_name:
                parameters[standard_name] = value
                break
    
    # ✅ Check if no valid values were extracted
    if not parameters:
        st.warning("⚠️ No valid numerical values detected! Please correct the extracted text manually or upload a clearer report.")
    
    return parameters


import pandas as pd
import os
from transformers import pipeline

# Load Zero-Shot Classification Model
medical_ai = pipeline("zero-shot-classification", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")


# Load past medical reports
db_file = "diagnosed.csv"
if os.path.exists(db_file):
    past_reports = pd.read_csv(db_file)
else:
    past_reports = pd.DataFrame(columns=["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT", "Diagnosis"])

# Function to predict disease
def predict_disease(parameters):
    global past_reports

    # Convert extracted values into a DataFrame-like structure
    report_df = pd.DataFrame([parameters])

    # Ensure column names match
    past_reports.columns = past_reports.columns.str.lower()
    report_df.columns = report_df.columns.str.lower()

    # Ensure "Diagnosis" column exists
    if "Diagnosis" not in past_reports.columns:
        past_reports["Diagnosis"] = None  # Create empty Diagnosis column

    # Find common columns
    common_cols = list(set(parameters.keys()) & set(past_reports.columns))

    # Ensure report_df and past_reports have the same column order
    report_df = report_df[common_cols]  
    past_reports = past_reports[common_cols + ["Diagnosis"]]  

    # Reset index to avoid mismatches
    past_reports = past_reports.reset_index(drop=True)
    report_df = report_df.reset_index(drop=True)

    # Compare reports (only on common columns)
    if not common_cols:
        return "⚠️ No matching columns found in the dataset. Using AI prediction."

       
    min_matching_cols = max(1, len(common_cols) // 2)  # Match at least half of the columns

    match = past_reports[
        past_reports[common_cols].apply(lambda row: (row == report_df.iloc[0][common_cols]).sum() >= min_matching_cols, axis=1)
    ]

    if not match.empty:
        return f"📌 Matched Previous Case: {match.iloc[0]['Diagnosis']}"


    if not match.empty:
        return f"📌 Matched Previous Case: {match.iloc[0]['Diagnosis']}"

    # If no match, use AI for classification
    report_text = "\n".join([f"{k}: {v}" for k, v in parameters.items()])
    prompt = f"""A patient's lab report contains the following test results:

{report_text}

Based on these lab values, what potential medical conditions could this patient have?"""

    labels = ["Anemia", "Leukemia", "Iron Deficiency", "Polycythemia", "Diabetes", "Thyroid Disorder", "Healthy"]
    prediction = medical_ai(prompt, candidate_labels=labels)
    diagnosis = prediction["labels"][0]  # Most likely condition

    # Save new report to the CSV file
    new_entry = report_df.copy()
    new_entry["Diagnosis"] = diagnosis
    past_reports = pd.concat([past_reports, new_entry], ignore_index=True)
    past_reports.to_csv(db_file, index=False)

    return f"🔬 AI-Based Prediction: {diagnosis}"


def main():
    st.title("Medical Report Analysis with AI")
    report_type = st.selectbox("Select Report Type", ["CBC", "Lipid Profile", "Thyroid Panel", "Diabetes Panel"])
    uploaded_files = st.file_uploader(f"Upload {report_type} report images or PDFs", type=["jpg", "jpeg", "png", "bmp", "pdf"], accept_multiple_files=True)
    
    all_parameters = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
        else:
            images = [Image.open(uploaded_file)]
        
        for image in images:
            processed_image = preprocess_image(image)
            result = ocr.ocr(processed_image, cls=True)
            json_data = save_ocr_results_to_json(result)
            parameters = extract_values_from_json(json_data)
            if parameters:
                all_parameters.append(parameters)
    
    if all_parameters:
        st.write("### Extracted Parameters")
        for i, parameters in enumerate(all_parameters):
            st.write(f"#### Report {i+1}")
            st.json(parameters)
            st.write(predict_disease(parameters))
    
if __name__ == "__main__":
    main()
