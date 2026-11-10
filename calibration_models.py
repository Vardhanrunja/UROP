import numpy as np
from scipy.special import softmax


class TemperatureScaledModel:
    def __init__(self, estimator, temperature):
        self.estimator = estimator
        self.temperature = float(temperature)
        self.classes_ = np.asarray(
            self.estimator.named_steps["classifier"].classes_
        )

    def _margins(self, X):
        imputer = self.estimator.named_steps["imputer"]
        classifier = self.estimator.named_steps["classifier"]
        transformed = imputer.transform(X)
        margins = classifier.predict(
            transformed,
            output_margin=True,
        )
        margins = np.asarray(margins)
        if margins.ndim != 2 or margins.shape[1] != len(self.classes_):
            raise ValueError(
                "XGBoost returned unexpected multiclass margins: "
                f"{margins.shape}"
            )
        return margins

    def predict_proba(self, X):
        return softmax(self._margins(X) / self.temperature, axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
