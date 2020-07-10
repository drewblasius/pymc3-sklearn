"""
ColumnTransformer that is pd.DataFrame-oriented.
"""

from sklearn.compose import ColumnTransformer
from typing import List, Union


class FrameTransformer(ColumnTransformer):
    """
    Modified version of `ColumnTransformer`, oriented towards dataframes rather than arrays.
    """

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass
