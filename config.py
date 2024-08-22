import os
from dotenv import load_dotenv

class Config(object):

    load_dotenv()
    SLIDER_CATEGORY_LABELS = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    CATEGORIES = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    TITLE = "Breast Cancer Predictor"
    
    DETAILS = """
        This application uses a machine learning model to predict whether a breast mass is benign or malignant based on cytology lab measurements. 
        Users can input the measurements by adjusting them manually using intuitive sliders in the sidebar, providing real-time prediction updates. 
        Tailored for healthcare professionals, the app enhances diagnostic accuracy and decision-making by showing how different cytological measurements affect the model's output.
    """

    DISCLAIMER = "This application is a data science project intended for educational purposes only. It is not designed to provide professional medical advice, diagnosis, or treatment. Healthcare professionals should not rely on this tool for clinical decision-making, and users should consult a qualified healthcare provider for any medical concerns."
    
    MODEL_FILE = os.getenv('MODEL_FILE')
    SCALER_FILE = os.getenv('SCALER_FILE')
    MODEL_DIR = os.getenv('MODEL_DIR')
    DATA_FILE = os.getenv('DATA_FILE')
    