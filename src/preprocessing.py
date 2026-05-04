"""
Препроцессинг для UCI Default of Credit Card Clients.

Идея:
- собираем ColumnTransformer один раз и используем его внутри Pipeline
  для обеих моделей (LogReg v1 и GradientBoosting v2);
- благодаря этому на этапе инференса в Flask API нам не нужно дублировать
  логику препроцессинга — модель уже знает, что делать с входным DataFrame.

Решения по фичам:
- ID убирается до подачи в препроцессор (это делает train.py).
- В исходном датасете в EDUCATION встречаются значения 0, 5, 6, которых
  нет в кодбуке UCI; в MARRIAGE — значение 0. Маппим их в "other"-категорию,
  чтобы не получить мусорные one-hot ветки на инференсе.
- SEX, EDUCATION, MARRIAGE, PAY_* — категориальные (PAY_* — статусы платежа,
  это дискретная порядковая шкала -2..9). Чтобы не плодить колонки и не ломать
  LogReg, кодируем их One-Hot только для социально-демографических полей.
  PAY_* оставляем как числовые: их значения упорядочены по тяжести просрочки.
- Числовые масштабируются StandardScaler-ом: нужно для LogReg, нейтрально
  для GBM.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Имена колонок ровно как в исходном CSV
TARGET_COLUMN = "default.payment.next.month"
ID_COLUMN = "ID"

CATEGORICAL_FEATURES: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_FEATURES: List[str] = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Все ожидаемые признаки на входе API (порядок важен для DataFrame)
EXPECTED_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Привести EDUCATION/MARRIAGE к значениям из кодбука UCI.

    Кодбук:
    - EDUCATION: 1=graduate school, 2=university, 3=high school, 4=others.
      Значения 0, 5, 6 встречаются в данных, но не задокументированы — маппим в 4.
    - MARRIAGE: 1=married, 2=single, 3=others. Значение 0 -> 3.
    """
    df = df.copy()
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def build_preprocessor() -> ColumnTransformer:
    """Собрать ColumnTransformer для смешанных типов признаков."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    # handle_unknown='ignore' защищает от категорий, которых не было на трейне,
    # — на инференсе такие просто кодируются нулевым вектором.
    categorical_pipeline = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Разделить на X/y и удалить ID, если он есть."""
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in dataframe. "
            f"Got columns: {list(df.columns)}"
        )
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    X = clean_categoricals(X)
    # Оставляем только ожидаемые колонки и фиксируем порядок
    X = X[EXPECTED_FEATURES]
    return X, y


def coerce_features(payload: dict) -> pd.DataFrame:
    """Превратить dict из JSON-запроса в одну строку DataFrame с нужным порядком колонок.

    Используется в Flask API. Числовые приводим к float, категориальные к int
    (в датасете SEX/EDUCATION/MARRIAGE — целочисленные коды).
    """
    row = {}
    for col in NUMERIC_FEATURES:
        row[col] = float(payload[col])
    for col in CATEGORICAL_FEATURES:
        row[col] = int(payload[col])
    df = pd.DataFrame([row], columns=EXPECTED_FEATURES)
    df = clean_categoricals(df)
    return df
