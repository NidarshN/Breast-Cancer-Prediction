import os
import pandas as pd
import streamlit as st
import pickle
from plotly import graph_objects as go
import numpy as np
from model.main import get_processed_data
from model.main import get_model
from config import Config
import time


class Cancer_Model():
    def __init__(self):
        if (Config.MODEL_FILE not in os.listdir(Config.MODEL_DIR)):
            print(Config.MODEL_FILE, Config.SCALER_FILE)
            get_model()

        self.model = pickle.load(open(Config.MODEL_DIR + Config.MODEL_FILE, "rb"))
        self.std_scalar = pickle.load(open(Config.MODEL_DIR + Config.SCALER_FILE, "rb"))

    def get_cancer_model(self):
        return self.model

    def get_std_scalar(self):
        return self.std_scalar


def create_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_processed_data()
    input_dict = {}

    for slider_label, key in Config.SLIDER_CATEGORY_LABELS:
        input_dict[key] = st.sidebar.slider(
            slider_label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


def get_normalized_data(input_dict):
    data = get_processed_data()
    X = data.drop(['diagnosis'], axis=1)

    normalized_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        norm_value = (value - min_value) / (max_value - min_value)
        normalized_dict[key] = norm_value

    return normalized_dict


def get_radar_chart(input_dict):
    norm_data = get_normalized_data(input_dict)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            norm_data['radius_mean'], norm_data['texture_mean'], norm_data['perimeter_mean'],
            norm_data['area_mean'], norm_data['smoothness_mean'], norm_data['compactness_mean'],
            norm_data['concavity_mean'], norm_data['concave points_mean'], norm_data['symmetry_mean'],
            norm_data['fractal_dimension_mean']
        ],
        theta=Config.CATEGORIES,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            norm_data['radius_se'], norm_data['texture_se'], norm_data['perimeter_se'], norm_data['area_se'],
            norm_data['smoothness_se'], norm_data['compactness_se'], norm_data['concavity_se'],
            norm_data['concave points_se'], norm_data['symmetry_se'], norm_data['fractal_dimension_se']
        ],
        theta=Config.CATEGORIES,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            norm_data['radius_worst'], norm_data['texture_worst'], norm_data['perimeter_worst'],
            norm_data['area_worst'], norm_data['smoothness_worst'], norm_data['compactness_worst'],
            norm_data['concavity_worst'], norm_data['concave points_worst'], norm_data['symmetry_worst'],
            norm_data['fractal_dimension_worst']
        ],
        theta=Config.CATEGORIES,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
    )

    return fig


def embed_predictions(input_data, model, std_scaler):
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    scaled_input = std_scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    st.subheader(f":blue[{'Cell Cluster Prediction'}]")
    st.write("The Cell Cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>",
                unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    st.write(f"Probability of being benign: :green[{round(model.predict_proba(scaled_input)[0][0] * 100, 2)} %]")
    st.write(f"Probability of being malicious: :red[{round(model.predict_proba(scaled_input)[0][1] * 100, 2)} %]")






def main():
    cancer_model_obj =  Cancer_Model()
    model = cancer_model_obj.get_cancer_model()
    std_scaler = cancer_model_obj.get_std_scalar()

    st.set_page_config(page_title=Config.TITLE, 
                    page_icon="",
                    layout="wide",
                    initial_sidebar_state="expanded",
                    menu_items={"About": Config.DISCLAIMER})
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = create_sidebar()
    with st.container():
        st.title(f":blue[{Config.TITLE}]")
        st.write(Config.DETAILS)
    column_graph, column_result = st.columns([4,1])

    with column_graph:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with column_result:
        embed_predictions(input_data, model, std_scaler)

    st.write(f":red[Disclaimer:]", Config.DISCLAIMER)


if __name__ == '__main__':
    main()