import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────
# DATA GENERATOR
# ─────────────────────────────
class MobileDataGenerator:
    def __init__(self, n_samples=5000, seed=42):
        self.rng = np.random.RandomState(seed)
        self.n_samples = n_samples

    def generate(self):
        r = self.rng
        n = self.n_samples

        df = pd.DataFrame({
            "transaction_amount": r.lognormal(4, 1.2, n),
            "transaction_hour": r.randint(0, 24, n),
            "transactions_per_day": r.poisson(4, n),
            "avg_transaction_amount": r.lognormal(3.5, 1.0, n),

            "device_age_days": r.randint(1, 1500, n),
            "new_device": r.binomial(1, 0.1, n),
            "location_change": r.binomial(1, 0.1, n),
            "distance_from_home_km": r.exponential(10, n),
            "vpn_usage": r.binomial(1, 0.05, n),

            "session_duration_sec": r.lognormal(5, 1, n),
            "login_attempts": r.poisson(1, n),
            "failed_logins": r.poisson(0.2, n),

            "typing_speed_wpm": r.normal(50, 10, n),
            "api_calls_per_minute": r.poisson(6, n),

            "network_type": r.choice(["wifi", "4g", "5g", "3g"], n),

            "account_age_days": r.randint(30, 2000, n),

            "is_fraud": r.binomial(1, 0.05, n)
        })

        return df


# ─────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────
class FeatureEngineer:
    def transform(self, df):
        df = df.copy()

        le = LabelEncoder()
        df["network_type"] = le.fit_transform(df["network_type"])

        df["risk_score"] = (
            df["new_device"] +
            df["location_change"] +
            df["vpn_usage"] +
            (df["failed_logins"] > 2).astype(int)
        )

        return df


# ─────────────────────────────
# MODEL
# ─────────────────────────────
class FraudClassifier:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        self.features = None

    def fit(self, X, y):
        self.features = X.columns
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def feature_importance(self):
        imp = self.model.named_steps["rf"].feature_importances_
        return pd.DataFrame({
            "feature": self.features,
            "importance": imp
        }).sort_values("importance", ascending=False)