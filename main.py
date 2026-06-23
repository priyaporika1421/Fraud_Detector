from Fraud_Detector import MobileDataGenerator, FeatureEngineer, FraudClassifier


class FraudDetectionPipeline:
    def __init__(self):
        self.gen = MobileDataGenerator()
        self.fe = FeatureEngineer()
        self.model = FraudClassifier()

    def train(self):
        df = self.gen.generate()

        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"]

        X = self.fe.transform(X)
        self.model.fit(X, y)

        print("✔ Model trained successfully")

    def analyse_transaction(self, data):
        import pandas as pd

        df = pd.DataFrame([data])
        df = self.fe.transform(df)

        prob = self.model.predict_proba(df)[0]

        return {
            "fraud_probability": float(round(prob * 100, 2)),
            "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        }


def main():
    pipeline = FraudDetectionPipeline()
    pipeline.train()

    test_data = {
        "transaction_amount": 9000,
        "transaction_hour": 2,
        "transactions_per_day": 15,
        "avg_transaction_amount": 8000,
        "device_age_days": 2,
        "new_device": 1,
        "location_change": 1,
        "distance_from_home_km": 300,
        "vpn_usage": 1,
        "session_duration_sec": 40,
        "login_attempts": 6,
        "failed_logins": 4,
        "typing_speed_wpm": 90,
        "api_calls_per_minute": 50,
        "network_type": "4g",
        "account_age_days": 10
    }

    print(pipeline.analyse_transaction(test_data))


if __name__ == "__main__":
    main()