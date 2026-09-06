import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Live Edge Traffic Prediction", page_icon="🚦", layout="wide")

# Representative road points for a city. TomTom returns live conditions for the
# road segment closest to each point, so these are samples rather than a
# city-wide vehicle count.
CITIES = {
    "Visakhapatnam": [
        (17.6868, 83.2185),
        (17.7041, 83.2977),
        (17.7215, 83.3012),
    ],
    "Vijayawada": [
        (16.5062, 80.6480),
        (16.5193, 80.6305),
        (16.4922, 80.6200),
    ],
    "Hyderabad": [
        (17.4435, 78.3772),
        (17.3850, 78.4867),
        (17.4483, 78.3915),
    ],
    "Chennai": [
        (13.0827, 80.2707),
        (13.0569, 80.2425),
        (13.0475, 80.2090),
    ],
    "Bengaluru": [
        (12.9716, 77.5946),
        (12.9352, 77.6245),
        (13.0358, 77.5970),
    ],
}


def get_secret(name: str):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


TOMTOM_API_KEY = get_secret("TOMTOM_API_KEY")


def fetch_flow(lat, lon):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}",
        "unit": "kmph",
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    flow = response.json().get("flowSegmentData", {})
    current = float(flow["currentSpeed"])
    free = float(flow["freeFlowSpeed"])
    current_tt = float(flow.get("currentTravelTime", 0))
    free_tt = float(flow.get("freeFlowTravelTime", 0))
    confidence = float(flow.get("confidence", 0))

    speed_ratio = current / free if free > 0 else 1.0
    congestion = max(0.0, min(1.0, 1.0 - speed_ratio))
    delay_seconds = max(0.0, current_tt - free_tt)

    if speed_ratio >= 0.80:
        status = "Low traffic"
    elif speed_ratio >= 0.55:
        status = "Moderate traffic"
    else:
        status = "Heavy traffic"

    return {
        "current_speed": current,
        "free_flow_speed": free,
        "speed_ratio": speed_ratio,
        "congestion": congestion,
        "delay_seconds": delay_seconds,
        "confidence": confidence,
        "status": status,
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_city_traffic(city):
    samples = []
    for lat, lon in CITIES[city]:
        try:
            item = fetch_flow(lat, lon)
            item["latitude"] = lat
            item["longitude"] = lon
            samples.append(item)
        except Exception as exc:
            samples.append({"error": str(exc), "latitude": lat, "longitude": lon})

    good = [x for x in samples if "error" not in x]
    if not good:
        raise RuntimeError("No live traffic sample could be retrieved for this city.")

    return good


st.title("🚦 Live Edge-Based Traffic Flow Prediction")
st.caption("Real-time traffic conditions + Lightweight LSTM + Knowledge Distillation + TensorFlow Lite")

city = st.sidebar.selectbox("📍 Select City", list(CITIES.keys()))
refresh = st.sidebar.button("🔄 Refresh Live Traffic", type="primary")

if refresh:
    get_city_traffic.clear()

if not TOMTOM_API_KEY:
    st.error("TomTom API key is not configured.")
    st.info(
        "For local testing create .streamlit/secrets.toml with "
        '\nTOMTOM_API_KEY = "YOUR_KEY"\n'
        "or set TOMTOM_API_KEY as an environment variable."
    )
    st.stop()

try:
    live = get_city_traffic(city)
except requests.HTTPError as exc:
    st.error(f"Traffic API request failed: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unable to get live traffic: {exc}")
    st.stop()

avg_speed = float(np.mean([x["current_speed"] for x in live]))
avg_free = float(np.mean([x["free_flow_speed"] for x in live]))
avg_congestion = max(0.0, min(1.0, 1.0 - avg_speed / avg_free)) if avg_free else 0.0
avg_delay = float(np.mean([x["delay_seconds"] for x in live]))
avg_confidence = float(np.mean([x["confidence"] for x in live]))

if avg_congestion < 0.20:
    status = "🟢 LOW"
elif avg_congestion < 0.45:
    status = "🟠 MODERATE"
else:
    status = "🔴 HEAVY"

st.subheader(f"📍 {city} — Live Traffic")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Avg Speed", f"{avg_speed:.1f} km/h")
c2.metric("Free-flow Avg Speed", f"{avg_free:.1f} km/h")
c3.metric("Congestion", f"{avg_congestion * 100:.0f}%")
c4.metric("Traffic Status", status)

st.caption(
    f"Live samples: {len(live)} road segments • API refresh: up to every 60 seconds • "
    f"Average confidence: {avg_confidence * 100:.0f}% • Approx. delay: {avg_delay:.0f} sec"
)

st.subheader("🛣️ Live Road Samples")
road_df = pd.DataFrame(
    [
        {
            "Sample": i + 1,
            "Current Speed (km/h)": round(x["current_speed"], 1),
            "Free-flow Speed (km/h)": round(x["free_flow_speed"], 1),
            "Congestion": f"{x['congestion'] * 100:.0f}%",
            "Status": x["status"],
            "Confidence": f"{x['confidence'] * 100:.0f}%",
        }
        for i, x in enumerate(live)
    ]
)
st.dataframe(road_df, use_container_width=True, hide_index=True)

st.divider()

# Existing distilled model section
if not os.path.exists("models/student_distilled.keras"):
    st.warning("Student model not found. Run `python train.py` first to enable AI prediction.")
    st.stop()

student = tf.keras.models.load_model("models/student_distilled.keras")
with open("results/metrics.json") as f:
    metrics = json.load(f)

st.subheader("🤖 Knowledge-Distillation Prediction")
st.caption("The live API reports road speed/congestion. The distilled LSTM predicts traffic flow from the project's trained traffic-flow sequence.")

# Keep the original model input flow available, but prefill from the project's dataset.
df = pd.read_csv("data/traffic.csv")
scaler = MinMaxScaler()
scaler.fit(df[["traffic_flow"]].iloc[: int(len(df) * 0.8)])

recent = df["traffic_flow"].tail(5).astype(float).tolist()
cols = st.columns(5)
values = []
for i, col in enumerate(cols):
    values.append(
        col.number_input(
            f"Time {i + 1}",
            min_value=0.0,
            value=float(recent[i]),
            step=1.0,
            key=f"live_ts_{i}",
        )
    )

if st.button("Predict Traffic Flow", type="primary"):
    x = scaler.transform(np.array(values).reshape(-1, 1)).reshape(1, 5, 1).astype("float32")
    pred = float(scaler.inverse_transform(student.predict(x, verbose=0))[0, 0])
    st.success(f"AI Predicted Traffic Flow: **{pred:.0f} vehicles**")

c1, c2, c3 = st.columns(3)
c1.metric("Teacher Parameters", f"{metrics['teacher']['parameters']:,}")
c2.metric("Student Parameters", f"{metrics['student']['parameters']:,}")
c3.metric("TFLite Size", f"{metrics['student_tflite_size_kb']:.1f} KB")

st.divider()
st.subheader("Teacher vs Student")
st.dataframe(
    pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE"],
            "Teacher": [metrics["teacher"]["mae_normalized"], metrics["teacher"]["rmse_normalized"]],
            "Student": [metrics["student"]["mae_normalized"], metrics["student"]["rmse_normalized"]],
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if os.path.exists("results/predictions.png"):
    st.subheader("Prediction Comparison")
    st.image("results/predictions.png")
if os.path.exists("results/model_comparison.png"):
    st.subheader("Model Parameter Comparison")
    st.image("results/model_comparison.png")

st.info(
    "Live traffic is measured from real-time road-speed data. It is not an exact city-wide vehicle count. "
    "For exact vehicle counts, camera/sensor volume data would be required."
)
