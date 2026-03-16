# Hybrid Solar–Biomass Energy Management System

## Smart Mini-Grid Control Interface

This project presents an AI-driven Hybrid Solar–Biomass Energy Management System built with Python and Streamlit. The system is designed to support smart mini-grid operation by predicting solar power generation, forecasting electricity demand, and dispatching biomass backup power whenever solar energy is insufficient.

## Project Overview

The system combines machine learning and optimization to improve energy reliability in a hybrid renewable energy setup. It performs three major tasks:

1. Predicts solar power generation from weather and time-related conditions
2. Forecasts electricity demand
3. Optimizes biomass dispatch to cover the energy gap when solar supply is not enough

The final application is deployed as an interactive Streamlit dashboard.

## Features

- Solar power prediction using a trained machine learning model
- Community electricity demand forecasting using a trained machine learning model
- Biomass dispatch optimization using linear programming
- Interactive Streamlit dashboard
- Clean and user-friendly interface
- Energy balance visualization with charts and gauges

## Models Used

### 1. Solar Forecasting Model
A machine learning regression model was trained to predict solar power generation based on:
- solar irradiance
- temperature
- humidity
- wind speed
- cloud cover
- time-based features

### 2. Demand Forecasting Model
A machine learning regression model was trained to forecast electricity demand using:
- weather variables
- time features
- historical load features
- rolling averages

### 3. Optimization Model
A linear programming model was used to determine:
- how much biomass power should be dispatched
- whether any energy remains unserved
- whether there is curtailment

## Folder Structure

```text
Hybrid-Solar-Biomass-App/
│
├── app.py
├── requirements.txt
├── README.md
│
└── saved_models/
    ├── solar_model.joblib
    ├── demand_model.joblib
    └── model_metadata.json
