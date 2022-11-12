import csv
import numpy as np
import pandas as pd
import gradio as gr
import joblib as jb
import os.path as path

import warnings

warnings.filterwarnings("ignore")

# Loading features names
file_csv = path.join("traffic_accidents", "model", "model_features.csv")
with open(file_csv) as f:
    reader = csv.reader(f)
    data = list(reader)

features = data[0]

# Creating a list with accident types
accident_type_list = [None,
                      "type_ATROPELAMENTO",
                      "type_CHOQUE",
                      "type_COLISÃO",
                      "type_OUTROS"]

# Loading the scaler
file_scaler_feridos = path.join("traffic_accidents", "model", "scaler_feridos.pkl")
scaler_feridos = jb.load(file_scaler_feridos)

# Loading the model
file_model_feridos = path.join("traffic_accidents", "model", "model_feridos.pkl")
model_feridos = jb.load(file_model_feridos)


def fit_inputs_injured(latitude,
                       longitude,
                       caminhao,
                       moto,
                       cars,
                       transport,
                       others,
                       holiday,
                       week_day,
                       hour_day,
                       accident_type) -> np.array:
    """This function will process data input
    from use to use in the model"""
    input_dict = {col: False for col in features}

    input_dict["latitude"] = latitude
    input_dict["longitude"] = longitude
    input_dict["caminhao"] = caminhao
    input_dict["moto"] = moto
    input_dict["cars"] = cars
    input_dict["transport"] = transport
    input_dict["others"] = others
    input_dict["holiday"] = holiday

    if week_day != 0:
        input_dict["day_" + str(week_day)] = True

    if hour_day != 0:
        input_dict["hour_" + str(hour_day)] = True

    if accident_type != 0:
        input_dict[accident_type_list[accident_type]] = True

    input_series = pd.Series(input_dict)

    input_array = input_series.to_numpy().reshape(1, -1)

    input_scaled = scaler_feridos.transform(input_array)

    return input_scaled


def predict(
        latitude,
        longitude,
        caminhao,
        moto,
        cars,
        transport,
        others,
        holiday,
        week_day,
        hour_day,
        accident_type) -> dict:
    """This function will be call by gradio
    when on submit action."""

    input_to_predict = fit_inputs_injured(latitude,
                                          longitude,
                                          caminhao,
                                          moto,
                                          cars,
                                          transport,
                                          others,
                                          holiday,
                                          week_day,
                                          hour_day,
                                          accident_type)

    predic_injured = model_feridos.predict_proba(input_to_predict)

    return {"No": predic_injured[0][0], "Yes": predic_injured[0][1]}

# print(predict(-30.054, -51.196, 0, 0, 1, 0, 0, 0, 1, 1, 0))

# demo = gr.Interface(
#     fn=predict,
#     inputs=[gr.Slider(
#                 minimum=-31.054,
#                 maximum=-29.054,
#                 step=0.001,
#                 value=-30.054,
#                 label="Latitude"),
#             gr.Slider(
#                 minimum=-52.196,
#                 maximum=-50.196,
#                 step=0.001,
#                 value=-51.196,
#                 label="Longitude"),
#             gr.Checkbox(label="Trucks involved?"),
#             gr.Checkbox(label="Motorcycle involved?"),
#             gr.Checkbox(label="Cars involved?"),
#             gr.Checkbox(label="Bus involved?"),
#             gr.Checkbox(label="Other vehicle (i.e. scooter) involved?"),
#             gr.Checkbox(label="Is holiday?"),
#             gr.Radio(
#                 choices=["Sun", "Mon",
#                          "Tue", "Wed",
#                          "Thu", "Fri",
#                          "Sat"],
#                 type="index",
#                 label="Day of Week"),
#             gr.Slider(
#                 minimum=0,
#                 maximum=23,
#                 step=1,
#                 label="Hour"),
#             gr.Dropdown(
#                 choices=["Violent Collision",
#                          "Running over",
#                          "Shock",
#                          "Collision",
#                          "Other"],
#                 type="index",
#                 label="Accident type")],
#     outputs=gr.Label(
#         label="Are there people injured?"))
#
# demo.launch()
