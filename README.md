# Edge-Based Traffic Flow Prediction Using Lightweight Deep Learning

A complete beginner-friendly demo project using an LSTM Teacher, lightweight LSTM Student, Knowledge Distillation, TensorFlow Lite, and Streamlit.

## Run on Windows
1. Install Python 3.10 or 3.11.
2. Extract this ZIP.
3. Open PowerShell in the extracted folder.
4. Run:
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   python train.py
   streamlit run app.py
5. Open http://localhost:8501 if the browser does not open automatically.

## One-click option
Double-click `run_project.bat`.

## Output
Training creates:
- models/teacher.keras
- models/student_distilled.keras
- models/student_model.tflite
- results/metrics.json
- results/predictions.png
- results/model_comparison.png

## Project flow
Traffic data -> preprocessing -> Teacher LSTM -> Student LSTM -> Knowledge Distillation -> evaluation -> TensorFlow Lite -> Streamlit dashboard.

The included CSV is synthetic demo data. Replace it with a real traffic dataset if your college/company requires one.

## Live traffic mode
The Streamlit dashboard can query TomTom Traffic Flow Segment Data for representative road points in each supported city. The API returns current speed, free-flow speed, travel time and confidence for the road segment closest to the requested coordinate.

Supported cities: Visakhapatnam, Vijayawada, Hyderabad, Chennai, Bengaluru.

### Local API key setup
1. Create a TomTom developer API key.
2. Create `.streamlit/secrets.toml`.
3. Add:

```toml
TOMTOM_API_KEY = "YOUR_TOMTOM_API_KEY"
```

4. Run `streamlit run app.py`.

Do not commit `.streamlit/secrets.toml` to GitHub. On Streamlit Community Cloud, add the same secret in the app's Secrets/Advanced settings.

Note: TomTom Flow Segment Data gives live road-segment speed/flow conditions, not an exact city-wide vehicle count. The dashboard therefore reports live speed and a congestion index. The project's LSTM remains a separate traffic-flow prediction model.
