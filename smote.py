from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval
import numbers
from collections import Counter

class DynamicSMOTE(BaseSampler):
    _parameter_constraints = {
        "random_state": [None, numbers.Integral],
        "minimo_amostras": [Interval(numbers.Integral, 1, None, closed="left")],
        "quantidade_gerada": [Interval(numbers.Integral, 1, None, closed="left")],
        "k_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
        "sampling_strategy": [None],  # Compatibilidade com BaseSampler
    }

    def __init__(self, random_state=None, minimo_amostras=5, quantidade_gerada=10, k_neighbors=5, sampling_strategy=None):
        self.random_state = random_state
        self.minimo_amostras = minimo_amostras
        self.quantidade_gerada = quantidade_gerada
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self._sampling_type = "over-sampling"  # Necessário para BaseSampler
        super().__init__(sampling_strategy=sampling_strategy)

    def _fit_resample(self, X, y):
        class_counts = Counter(y)

        if not self.minimo_amostras or not self.quantidade_gerada or not self.k_neighbors:
            print("SEM PARAMETROS")
            return X, y

        strategy = {
            cls: self.quantidade_gerada for cls, count in class_counts.items()
            if count <= self.minimo_amostras
        }

        print(strategy)

        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors
        )

        return smote.fit_resample(X, y)
