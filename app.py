# app.py
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, flash
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField, FloatField
from wtforms.validators import DataRequired, NumberRange

# --- Configuration ---
PIPELINE_FILE = "full_pipeline.joblib"
MODEL_FILE = "fires_model.keras"

# --- Load Artifacts ---
print("--- Loading Artifacts for Flask App ---")
# (Loading code - 이전과 동일)
try:
    pipeline = joblib.load(PIPELINE_FILE)
    print(f"Pipeline loaded successfully from {PIPELINE_FILE}")
except FileNotFoundError:
    print(f"Error: Pipeline file not found at {PIPELINE_FILE}.")
    pipeline = None
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

try:
    model = keras.models.load_model(MODEL_FILE)
    print(f"Model loaded successfully from {MODEL_FILE}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_FILE}.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-default-hard-to-guess-key-change-me')
bootstrap5 = Bootstrap5(app)

# --- Form Definition ---
# (Form definition code - 이전과 동일)
MONTH_CHOICES = [
    ('01-Jan', '01-Jan'), ('02-Feb', '02-Feb'), ('03-Mar', '03-Mar'),
    ('04-Apr', '04-Apr'), ('05-May', '05-May'), ('06-Jun', '06-Jun'),
    ('07-Jul', '07-Jul'), ('08-Aug', '08-Aug'), ('09-Sep', '09-Sep'),
    ('10-Oct', '10-Oct'), ('11-Nov', '11-Nov'), ('12-Dec', '12-Dec')
]
DAY_CHOICES = [
    ('00-sun', 'Sunday'), ('01-mon', 'Monday'), ('02-tue', 'Tuesday'),
    ('03-wed', 'Wednesday'), ('04-thu', 'Thursday'), ('05-fri', 'Friday'),
    ('06-sat', 'Saturday'), ('07-hol', 'Holiday')
]

class PredictionForm(FlaskForm):
    longitude = IntegerField('Longitude (격자 X좌표)',
                           validators=[DataRequired(), NumberRange(min=1, max=7, message="경도 격자값은 1~7 사이여야 합니다.")],
                           description="1 ~ 7 사이의 정수")
    latitude = IntegerField('Latitude (격자 Y좌표)',
                          validators=[DataRequired(), NumberRange(min=1, max=7, message="위도 격자값은 1~7 사이여야 합니다.")],
                          description="1 ~ 7 사이의 정수")
    month = SelectField('Month', choices=MONTH_CHOICES, validators=[DataRequired()])
    day = SelectField('Day of Week', choices=DAY_CHOICES, validators=[DataRequired()])
    avg_temp = FloatField('Average Temperature (°C)', validators=[DataRequired()], description="예: 15.5")
    max_temp = FloatField('Max Temperature (°C)', validators=[DataRequired()], description="예: 25.0")
    max_wind_speed = FloatField('Max Wind Speed (m/s)', validators=[DataRequired(), NumberRange(min=0)], description="예: 8.5")
    avg_wind = FloatField('Average Wind Speed (m/s)', validators=[DataRequired(), NumberRange(min=0)], description="예: 3.2")
    submit = SubmitField('Predict Burned Area')


# --- Flask Routes ---
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form = PredictionForm()
    prediction_result_text = None

    if form.validate_on_submit():
        if pipeline is None or model is None:
            flash("오류: 모델 또는 파이프라인이 로드되지 않았습니다.", "danger")
            return render_template('prediction.html', form=form)

        try:
            input_data = {
                'longitude': [form.longitude.data], 'latitude': [form.latitude.data],
                'month': [form.month.data], 'day': [form.day.data],
                'avg_temp': [form.avg_temp.data], 'max_temp': [form.max_temp.data],
                'max_wind_speed': [form.max_wind_speed.data], 'avg_wind': [form.avg_wind.data]
            }
            input_df = pd.DataFrame(input_data)
            print(f"\nInput DataFrame:\n{input_df}")

            print("Transforming input data...")
            input_prepared = pipeline.transform(input_df)
            print(f"Prepared input shape: {input_prepared.shape}")

            print("Making prediction...")
            pred_log = model.predict(input_prepared)[0][0] # Get the scalar value
            print(f"Predicted value (log scale): {pred_log}")

            # --- 여기가 수정된 부분: 단위 변환 ---
            # 1. 역변환 (exp(x) - 1) -> CSV 스케일 값 복원
            pred_original_csv_scale = np.expm1(pred_log)
            pred_original_csv_scale = max(0, pred_original_csv_scale) # Ensure non-negative
            print(f"Predicted value (CSV scale): {pred_original_csv_scale}")

            # 2. 실제 면적(m^2) 계산 (CSV 값이 m^2/100 이라고 가정)
            predicted_area_m2 = pred_original_csv_scale * 100
            print(f"Predicted value (m^2 scale): {predicted_area_m2}")

            # 3. 결과 포맷팅 (m^2 단위 사용)
            unit = "m²" # PDF 예시와 동일하게 m^2 사용
            prediction_result_text = f"{predicted_area_m2:.2f} {unit}"
            # --- 여기까지 수정됨 ---

            return render_template('result.html', prediction=prediction_result_text)

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash(f"예측 중 오류가 발생했습니다. 입력값을 확인해주세요. 오류: {e}", "danger")
            return render_template('prediction.html', form=form)

    return render_template('prediction.html', form=form)

# --- Main Execution Block for Flask App ---
if __name__ == '__main__':
    if pipeline is None or model is None:
        print("*"*50)
        print(" ERROR: Model/Pipeline not loaded. Run 'python train.py' first.")
        print("*"*50)
    else:
        print("\n--- Starting Flask App ---")
        print("Access at http://127.0.0.1:5000 (or your local IP)")
        app.run(debug=True, host='0.0.0.0', port=5000)