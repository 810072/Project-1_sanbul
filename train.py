# train.py
# PDF ('Project#1 - AI기반 산불 예측 서비스 개발.pdf') 기반 재작성

import pandas as pd
import numpy as np
import joblib
import os

# Scikit-Learn imports (PDF에서 언급된 모듈 위주)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error # 평가를 위해 추가

# TensorFlow and Keras imports (PDF Step 2 기반)
import tensorflow as tf
from tensorflow import keras

# --- Configuration ---
DATA_FILE = "sanbul2district-divby100.csv" # PDF Page 3
PIPELINE_FILE = "full_pipeline.joblib"     # 저장 파일 (관례적 이름)
MODEL_FILE = "fires_model.keras"           # PDF Page 16

RANDOM_STATE = 42                          # PDF Page 11, 16

# --- 특성 목록 정의 (PDF Page 19 기반) ---
# 주의: PDF Page 14의 fires_num 정의는 코드 스니펫에 오타가 있을 수 있음
# ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'] 사용
NUMERICAL_FEATURES = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
CATEGORICAL_FEATURES = ['month', 'day']
TARGET_FEATURE = 'burned_area' # 원본 타겟 컬럼명

def train_and_save(data_path=DATA_FILE, pipeline_path=PIPELINE_FILE, model_path=MODEL_FILE):
    """
    PDF 단계에 따라 데이터 로드, 전처리, 모델 학습 및 저장을 수행합니다.
    """
    print(f"--- Starting Training Process based on PDF ---")

    # --- 단계 1: Data 전처리 ---

    # 1-1: Data 불러오기 (PDF Page 6)
    try:
        fires = pd.read_csv(data_path, sep=",")
        print(f"Step 1-1: Data loaded from {data_path}. Shape: {fires.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1-4: 특성 burned_area 로그 변환 (PDF Page 6, 10)
    # PDF Page 6 코드: fires['burned_area'] = np.log(fires['burned_area'] + 1)
    # np.log(x + 1)은 np.log1p(x)와 동일하며, 0값 처리에 더 안정적
    if TARGET_FEATURE not in fires.columns:
        print(f"Error: Target column '{TARGET_FEATURE}' not found.")
        return
    if (fires[TARGET_FEATURE] < 0).any():
         print(f"Warning: Negative values found in '{TARGET_FEATURE}'. Clamping to 0.")
         fires[TARGET_FEATURE] = fires[TARGET_FEATURE].clip(lower=0)
    # 로그 변환 후 원본 컬럼에 다시 저장 (PDF Page 6 방식)
    fires[TARGET_FEATURE] = np.log1p(fires[TARGET_FEATURE])
    print(f"Step 1-4: '{TARGET_FEATURE}' log transformed (using np.log1p) and updated in DataFrame.")

    # (Optional) 1-2: 기본 정보 확인 (PDF Page 8) - 필요시 주석 해제
    # print("\nStep 1-2: Basic Data Info")
    # print("Head:\n", fires.head())
    # print("\nInfo:")
    # fires.info()
    # print("\nDescribe:\n", fires.describe())
    # print("\nMonth Counts:\n", fires['month'].value_counts())
    # print("\nDay Counts:\n", fires['day'].value_counts())

    # 1-5: Training/Test Set 분리 (StratifiedShuffleSplit 사용) (PDF Page 11)
    if 'month' not in fires.columns:
        print("Error: 'month' column not found for stratified splitting.")
        return
    print(f"\nStep 1-5: Splitting data using StratifiedShuffleSplit (random_state={RANDOM_STATE})...")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for train_index, test_index in split.split(fires, fires["month"]):
        strat_train_set = fires.loc[train_index]
        strat_test_set = fires.loc[test_index]
    print(f"Training set size: {len(strat_train_set)}, Test set size: {len(strat_test_set)}")

    # 학습 데이터에서 레이블 분리 (PDF Page 14)
    # 주의: 이때 분리하는 'burned_area'는 이미 로그 변환된 값임
    fires_train_labels = strat_train_set[TARGET_FEATURE].copy()
    # PDF Page 14: fires = strat_train_set.drop(["burned_area"], axis=1)
    fires_train_features = strat_train_set.drop(TARGET_FEATURE, axis=1)
    print("Labels separated from training features.")

    # 테스트 데이터에서 레이블 분리
    fires_test_labels = strat_test_set[TARGET_FEATURE].copy()
    fires_test_features = strat_test_set.drop(TARGET_FEATURE, axis=1)
    print("Labels separated from test features.")

    # 1-9: Scikit-Learn Pipeline 구성 (PDF Page 15)
    # 수치형 파이프라인
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    # 전체 파이프라인 (ColumnTransformer)
    # PDF Page 19 에서 num_attribs, cat_attribs 정의 참고
    num_attribs = NUMERICAL_FEATURES
    cat_attribs = CATEGORICAL_FEATURES

    # ColumnTransformer 정의 (remainder='drop' 추가하여 명확성 확보)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs), # handle_unknown='ignore'는 PDF에 명시되지 않았으나 권장됨
    ], remainder='drop')
    print("\nStep 1-9: Preprocessing pipeline defined (StandardScaler for num, OneHotEncoder for cat).")

    # 파이프라인 학습 데이터에 적용 (fit_transform)
    print("Fitting and transforming training data...")
    # PDF Page 15: fires_prepared = full_pipeline.fit_transform(fires)
    # 여기서 fires는 레이블이 제거된 fires_train_features에 해당
    fires_train_prepared = full_pipeline.fit_transform(fires_train_features)
    print(f"Training data prepared shape: {fires_train_prepared.shape}")

    # 파이프라인 테스트 데이터에 적용 (transform)
    print("Transforming test data...")
    fires_test_prepared = full_pipeline.transform(fires_test_features)
    print(f"Test data prepared shape: {fires_test_prepared.shape}")

    # 파이프라인 저장
    print(f"\nSaving preprocessing pipeline to {pipeline_path}...")
    joblib.dump(full_pipeline, pipeline_path)
    print("Pipeline saved.")

    # --- 단계 2: Keras 모델 개발 (PDF Page 16) ---
    print("\n--- Step 2: Keras Model Development ---")

    # 검증 세트 분리 (Prepared된 학습 데이터 사용)
    # PDF Page 16: X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, ...)
    print(f"Splitting prepared training data into train/validation sets (random_state={RANDOM_STATE})...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        fires_train_prepared, fires_train_labels, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"X_train shape: {X_train.shape}, X_valid shape: {X_valid.shape}")

    # 재현성을 위한 시드 설정 (PDF Page 16)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    print(f"Random seeds set to {RANDOM_STATE}.")

    # Keras 모델 정의 (PDF Page 16)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
        keras.layers.Dense(30, activation='relu'), # PDF 코드에는 input_shape가 중복 명시되어 있으나, 두번째부터는 불필요
        keras.layers.Dense(30, activation='relu'),
        keras.layers.Dense(1) # 회귀 문제이므로 출력 노드 1개, 활성화 함수 없음
    ])
    print("\nKeras model defined:")
    model.summary()

    # 모델 컴파일 (PDF Page 16)
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(f"Model compiled with loss='mean_squared_error' and optimizer=SGD(lr=1e-3).")

    # 모델 학습 (PDF Page 16)
    print("\nTraining Keras model...")
    history = model.fit(X_train, y_train, epochs=200, # PDF 명시된 에포크
                        validation_data=(X_valid, y_valid), verbose=1) # verbose=1로 진행 상황 표시
    print("Model training finished.")

    # (Optional) 모델 평가 (PDF Page 16 하단 평가 부분 참고)
    print("\nEvaluating model on the test set...")
    # PDF Page 16: X_test, y_test = fires_test_prepared, fires_test_labels
    # 이 변수들은 이미 준비되어 있음
    test_loss = model.evaluate(fires_test_prepared, fires_test_labels, verbose=0)
    print(f"Test set Mean Squared Error (log scale): {test_loss:.4f}")
    # RMSE 계산
    test_rmse = np.sqrt(test_loss)
    print(f"Test set Root Mean Squared Error (log scale): {test_rmse:.4f}")

    # Keras 모델 저장 (PDF Page 16)
    print(f"\nSaving trained model to {model_path}...")
    model.save(model_path)
    print("Model saved.")
    print("--- Training Process Finished ---")


if __name__ == '__main__':
    # 기존 아티팩트 파일 삭제 (선택적)
    if os.path.exists(PIPELINE_FILE):
        print(f"Removing existing pipeline file: {PIPELINE_FILE}")
        os.remove(PIPELINE_FILE)
    if os.path.exists(MODEL_FILE):
        print(f"Removing existing model file: {MODEL_FILE}")
        os.remove(MODEL_FILE)

    # 학습 및 저장 함수 실행
    train_and_save()