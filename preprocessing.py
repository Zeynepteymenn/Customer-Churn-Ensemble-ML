"""
Veri yukleme ve on isleme fonksiyonlari.
Telco Customer Churn veri setine gore hazirlanmistir.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """CSV dosyasini yukler."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik/hatali degerleri temizler, gereksiz kolonlari kaldirir."""
    df = df.copy()

    # customerID modelleme icin anlamsiz, kaldirilir
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges bazen bosluk iceren string olarak gelir
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_features(df: pd.DataFrame):
    """Kategorik degiskenleri encode eder, hedef degiskeni ayirir."""
    df = df.copy()

    # Hedef degisken: Yes/No -> 1/0
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders


def split_and_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train/test bolme ve olceklendirme islemi."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_pipeline(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """Tum on isleme adimlarini calistirip train/test verisini dondurur."""
    df = load_data(path)
    df = clean_data(df)
    X, y, encoders = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )
    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, scaler, encoders, feature_names
