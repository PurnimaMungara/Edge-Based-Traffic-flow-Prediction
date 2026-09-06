import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_preprocessing import load_and_prepare_data
from src.models import build_teacher, build_student
from src.distillation import Distiller


# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n======================================")
print("        LOADING TRAFFIC DATA")
print("======================================")

X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
    "data/traffic.csv",
    5,
    0.2
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# TEACHER
# ============================================================

print("\n======================================")
print("          BUILDING TEACHER")
print("======================================")

teacher = build_teacher(
    X_train.shape[1],
    X_train.shape[2]
)

teacher.summary()

teacher.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


print("\n======================================")
print("          TRAINING TEACHER")
print("======================================")

teacher.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# Save teacher
teacher_path = "models/teacher.keras"

teacher.save(teacher_path)

print("\nTeacher saved:")
print(teacher_path)


# ============================================================
# STUDENT
# ============================================================

print("\n======================================")
print("           BUILDING STUDENT")
print("======================================")

student = build_student(
    X_train.shape[1],
    X_train.shape[2]
)

student.summary()


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

print("\n======================================")
print("        KNOWLEDGE DISTILLATION")
print("======================================")

distiller = Distiller(
    student,
    teacher,
    alpha=0.5
)

distiller.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
)


print("\n======================================")
print("        TRAINING STUDENT")
print("======================================")

distiller.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# Save student
student_path = "models/student_distilled.keras"

student.save(student_path)

print("\nStudent saved:")
print(student_path)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n======================================")
print("         EVALUATING MODELS")
print("======================================")

start = time.perf_counter()

teacher_pred = teacher.predict(
    X_test,
    verbose=0
)

teacher_total_time = time.perf_counter() - start

teacher_time = teacher_total_time / max(len(X_test), 1)


start = time.perf_counter()

student_pred = student.predict(
    X_test,
    verbose=0
)

student_total_time = time.perf_counter() - start

student_time = student_total_time / max(len(X_test), 1)


# ============================================================
# FLATTEN
# ============================================================

y_test_flat = np.asarray(y_test).reshape(-1)

teacher_pred_flat = np.asarray(
    teacher_pred
).reshape(-1)

student_pred_flat = np.asarray(
    student_pred
).reshape(-1)


# ============================================================
# METRICS
# ============================================================

teacher_mae = float(
    mean_absolute_error(
        y_test_flat,
        teacher_pred_flat
    )
)

student_mae = float(
    mean_absolute_error(
        y_test_flat,
        student_pred_flat
    )
)

teacher_rmse = float(
    np.sqrt(
        mean_squared_error(
            y_test_flat,
            teacher_pred_flat
        )
    )
)

student_rmse = float(
    np.sqrt(
        mean_squared_error(
            y_test_flat,
            student_pred_flat
        )
    )
)


# ============================================================
# TFLITE
# ============================================================

print("\n======================================")
print("        CREATING TFLITE MODEL")
print("======================================")

tflite_success = False
tflite_size_kb = 0.0

try:

    converter = tf.lite.TFLiteConverter.from_keras_model(
        student
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Required for TensorList operations in LSTM
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    tflite_path = "models/student_model.tflite"

    with open(
        tflite_path,
        "wb"
    ) as f:
        f.write(tflite_model)

    tflite_size_kb = len(tflite_model) / 1024

    tflite_success = True

    print("\nTFLite model saved:")
    print(tflite_path)

    print(
        "TFLite size:",
        round(tflite_size_kb, 2),
        "KB"
    )

except Exception as e:

    print("\nWARNING:")
    print("TFLite conversion failed.")
    print("The trained Student .keras model is still available.")
    print("\nTFLite error:")
    print(e)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "teacher": {
        "mae_normalized": teacher_mae,
        "rmse_normalized": teacher_rmse,
        "parameters": int(
            teacher.count_params()
        ),
        "avg_prediction_seconds": teacher_time
    },

    "student": {
        "mae_normalized": student_mae,
        "rmse_normalized": student_rmse,
        "parameters": int(
            student.count_params()
        ),
        "avg_prediction_seconds": student_time
    },

    "student_tflite_size_kb": round(
        tflite_size_kb,
        2
    ),

    "tflite_conversion_success": tflite_success
}


with open(
    "results/metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_test_2d = np.asarray(
    y_test
).reshape(-1, 1)

teacher_pred_2d = np.asarray(
    teacher_pred
).reshape(-1, 1)

student_pred_2d = np.asarray(
    student_pred
).reshape(-1, 1)


yt = scaler.inverse_transform(
    y_test_2d
)

tp = scaler.inverse_transform(
    teacher_pred_2d
)

sp = scaler.inverse_transform(
    student_pred_2d
)


# ============================================================
# PREDICTION GRAPH
# ============================================================

n = min(
    100,
    len(yt)
)

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    yt[:n],
    label="Actual"
)

plt.plot(
    tp[:n],
    label="Teacher"
)

plt.plot(
    sp[:n],
    label="Student"
)

plt.title(
    "Traffic Flow: Actual vs Teacher vs Distilled Student"
)

plt.xlabel(
    "Test Time Step"
)

plt.ylabel(
    "Vehicles"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/predictions.png",
    dpi=150
)

plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    ["Teacher", "Student"],
    [
        teacher.count_params(),
        student.count_params()
    ]
)

plt.title(
    "Model Parameter Comparison"
)

plt.ylabel(
    "Number of Parameters"
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=150
)

plt.close()


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("             === RESULTS ===")
print("======================================")

print(
    "Teacher parameters:",
    teacher.count_params()
)

print(
    "Student parameters:",
    student.count_params()
)

print(
    "Teacher MAE:",
    teacher_mae
)

print(
    "Student MAE:",
    student_mae
)

print(
    "Teacher RMSE:",
    teacher_rmse
)

print(
    "Student RMSE:",
    student_rmse
)

print(
    "Teacher prediction time:",
    teacher_time,
    "seconds/sample"
)

print(
    "Student prediction time:",
    student_time,
    "seconds/sample"
)

print(
    "TFLite size:",
    round(
        tflite_size_kb,
        2
    ),
    "KB"
)


# ============================================================
# FILES
# ============================================================

print("\n======================================")
print("          GENERATED FILES")
print("======================================")

print("models/teacher.keras")
print("models/student_distilled.keras")

if tflite_success:
    print("models/student_model.tflite")
else:
    print("models/student_model.tflite -> NOT CREATED")

print("results/metrics.json")
print("results/predictions.png")
print("results/model_comparison.png")


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("               DONE")
print("======================================")

print("\nRun Streamlit using:")

print("streamlit run app.py")