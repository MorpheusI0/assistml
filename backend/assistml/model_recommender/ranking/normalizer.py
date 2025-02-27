from typing import List, Literal


class Normalizer:
    _optimization_goal: Literal["minimize", "maximize"]
    _fitted: bool
    _min_value: float
    _max_value: float

    def __init__(self, optimization_goal: Literal["minimize", "maximize"] = "maximize"):
        if optimization_goal not in ["minimize", "maximize"]:
            raise ValueError(f"Invalid optimization goal: {optimization_goal}")
        self._optimization_goal = optimization_goal
        self._fitted = False
        _min_value = None
        _max_value = None

    def fit(self, values: List[float]):
        if not values:
            raise ValueError("No values to fit")
        self._min_value = min(values)
        self._max_value = max(values)
        self._fitted = True

    def transform(self, value: float) -> float:
        if not self._fitted:
            raise ValueError("Normalizer is not fitted. Call fit() first.")
        if self._min_value == self._max_value:
            return 1.0  # a single value is always optimal
        if self._optimization_goal == "minimize":
            norm = (value - self._min_value) / (self._max_value - self._min_value)
        else:
            norm = 1 - (value - self._min_value) / (self._max_value - self._min_value)
        # in case of numerical inaccuracies clip the value to [0, 1]
        return max(0.0, min(1.0, norm))
