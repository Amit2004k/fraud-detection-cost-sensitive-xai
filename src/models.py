"""
models.py
---------
Cost-sensitive fraud detection model wrappers.
Supports XGBoost, LightGBM, RandomForest with asymmetric cost weighting.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
import xgboost as xgb


class CostSensitiveFraudDetector:
    """
    A cost-sensitive fraud detection wrapper.

    Parameters
    ----------
    model_type : str
        One of 'xgboost', 'random_forest', 'logistic'
    fn_cost : float
        Cost of a false negative (missed fraud). Default 100.
    fp_cost : float
        Cost of a false positive (false alarm). Default 1.
    """

    def __init__(self, model_type="xgboost", fn_cost=100, fp_cost=1):
        self.model_type = model_type
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.model = None
        self._scale_pos_weight = fn_cost / fp_cost

    def _build_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                scale_pos_weight=self._scale_pos_weight,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                random_state=42,
                use_label_encoder=False,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                class_weight={0: self.fp_cost, 1: self.fn_cost},
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                class_weight={0: self.fp_cost, 1: self.fn_cost},
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self._build_model()
        if self.model_type == "xgboost" and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def optimal_threshold(self, X_val, y_val):
        """Find threshold minimizing total financial cost on validation set."""
        probs = self.predict_proba(X_val)
        thresholds = np.linspace(0.01, 0.99, 200)
        best_t, best_cost = 0.5, float("inf")
        for t in thresholds:
            y_pred = (probs >= t).astype(int)
            fn = ((y_val == 1) & (y_pred == 0)).sum()
            fp = ((y_val == 0) & (y_pred == 1)).sum()
            cost = fn * self.fn_cost + fp * self.fp_cost
            if cost < best_cost:
                best_cost = cost
                best_t = t
        print(f"Optimal threshold: {best_t:.3f} | Min cost: {best_cost:,.0f}")
        return best_t

    def evaluate(self, X_test, y_test, threshold=0.5):
        probs = self.predict_proba(X_test)
        y_pred = (probs >= threshold).astype(int)
        pr_auc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, y_pred)
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        total_cost = fn * self.fn_cost + fp * self.fp_cost
        print(f"PR-AUC       : {pr_auc:.4f}")
        print(f"F1 (fraud)   : {f1:.4f}")
        print(f"False Neg    : {fn}  (missed fraud)")
        print(f"False Pos    : {fp}  (false alarms)")
        print(f"Total Cost   : {total_cost:,.0f} units")
        return {"pr_auc": pr_auc, "f1": f1, "fn": fn, "fp": fp, "cost": total_cost}
