import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class PredictiveMaintenance:

    def __init__(self):

        self.df = pd.read_csv('predictive_maintenance.csv')

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

        self.models = {}
        self.results = None
        self.best_model = None
        self.best_model_name = None

        self.preprocess()
        self.train_models()

    def preprocess(self):

        df = self.df.copy()

        df.drop(["UDI", "Product ID"], axis=1, inplace=True)

        df["Type"] = self.encoder.fit_transform(df["Type"])

        X = df.drop(["Target", "Failure Type"], axis=1)
        y = df["Target"]

        self.feature_names = X.columns

        X_scaled = self.scaler.fit_transform(X)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def train_models(self):

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "LightGBM": LGBMClassifier()
        }

        results = []

        for name, model in models.items():

            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)

            y_prob = model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc = roc_auc_score(self.y_test, y_prob)

            results.append([
                name,
                accuracy,
                precision,
                recall,
                f1,
                roc
            ])

            self.models[name] = model

        self.results = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
        )

        self.results = self.results.sort_values(
            by="F1 Score",
            ascending=False
        )

        self.best_model_name = self.results.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]

    def predict(self, values):

        data = pd.DataFrame(
            [values],
            columns=self.feature_names
        )

        data_scaled = self.scaler.transform(data)

        prediction = self.best_model.predict(data_scaled)[0]

        probability = self.best_model.predict_proba(
            data_scaled
        )[0][1]

        return prediction, probability

    def explain_failure(self, values):

        reasons = []

        if values[5] > 200:
            reasons.append(
                "Tool wear exceeds safe operating limit."
            )

        if values[4] > 60:
            reasons.append(
                "High torque causing excessive spindle load."
            )

        if values[2] > 315:
            reasons.append(
                "Process temperature is abnormally high."
            )

        if values[3] > 1800:
            reasons.append(
                "High rotational speed increasing stress."
            )

        return reasons

    def suggestions(self, values):

        suggestions = []

        if values[5] > 200:
            suggestions.append(
                "Replace cutting tool."
            )

        if values[4] > 60:
            suggestions.append(
                "Inspect spindle load and alignment."
            )

        if values[2] > 315:
            suggestions.append(
                "Check coolant circulation system."
            )

        if values[3] > 1800:
            suggestions.append(
                "Reduce machine operating speed."
            )

        if len(suggestions) == 0:
            suggestions.append(
                "Continue routine preventive maintenance."
            )

        return suggestions
