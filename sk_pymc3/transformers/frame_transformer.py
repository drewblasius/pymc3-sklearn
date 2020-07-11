"""
ColumnTransformer that is pd.DataFrame-oriented.
"""

import pandas as pd
import numpy as np

from copy import copy
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Tuple, Union


class PassthroughTransformer(TransformerMixin, BaseEstimator):

    def fit(self, y):
        return self

    def transform(self, y):
        return y


class FrameTransformer(TransformerMixin, BaseEstimator):
    """
    Modified version of `ColumnTransformer`, oriented towards dataframes rather than arrays.
    """

    def __init__(
            self, 
            transformers: List[Tuple[str, TransformerMixin, List[str]]],
            remainder: Optional[str] = "drop",
    ):

        if remainder not in ("passthrough", "drop"):
            raise ValueError("`remainder` must be one of ('passthrough', 'drop')")

        self.remainder = remainder
        self.transformers = transformers

    def _handle_passthrough(
        self, transformer: Union[str, TransformerMixin]
    ) -> TransformerMixin:
        if transformer == "passthrough":
            return PassthroughTransformer()
        return transformer

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None):
        
        self.fitted_transformers = []
        for name, transformer, cols in self.transformers:
            for col in cols:
                tr = copy(
                    self._handle_passthrough(transformer)
                )
                tr.fit(X[col].values.reshape(-1, 1))
                self.fitted_transformers.append(
                    (name, tr, col)
                )

        return self

    def transform(self, X, y=None):
        Xo = X.copy()
        for name, transformer, col in self.fitted_transformers:
            Xo[col] = transformer.transform(Xo[col].values.reshape(-1, 1))
        
        if self.remainder == "drop":
            Xo = Xo[
                [col for _, _, col in self.fitted_transformers]
            ]
        
        return Xo