"""
ColumnTransformer that is pd.DataFrame-oriented.
"""

from sklearn.compose import BaseEstimator, TransformerMixin
from typing import List, Optional, Union


class PassthroughTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        return X


class FrameTransformer(TransformerMixin, BaseEstimator):
    """
    Modified version of `ColumnTransformer`, oriented towards dataframes rather than arrays.
    """

    def __init__(
            self, 
            transformers: List[Tuple[str, TransformerMixin, List[str]]],
            remainder: Optional[str] = "passthrough")
    ):

        if remainder not in ("passthrough", "drop"):
            raise ValueError("`remainder` must be one of ('passthrough', 'drop')")

        self.transformers = transformers


    def _handle_passthrough(
            transformer: Union[str, TransformerMixin]
    ) -> TransformerMixin:
        if transformer == "passthrough":
            return PassthroughTransformer()
        return transformer

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None):
        
        self.fitted_transformers = []
        for name, transformer, cols in self.transformers:
            for col in cols:
                tr = self._handle_passthrough(transfomer).copy()
                tr.fit(X[col].values.reshape(-1, 1))

                self.fitted_transformers.append(
                    (name, tr, col)
                )

        return self

    def transform(self, X, y=None):
        Xo = X.copy()
        for name, transformer, col in self.fitted_transformers:
            Xo[col] = transformer.transform(Xo[col].values)
        
        if self.remainder == "drop":
            Xo = Xo[
                [col for _, _, col in self.fitted_transformers]
            ]
        
        return Xo