import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import theano as tt

from abc import ABC, abstractmethod
from contextlib import contextmanager
from sklearn.base import BaseEstimator
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BasePyMC3Model(ABC, BaseEstimator):

    @abstractmethod
    def model_block(self) -> pm.backends.base.MultiTrace:
        """
        Abstract method that contains all of the model information.

        *MUST* return a multi-trace object.
        """
        pass

    def fit_model(self) -> pm.backends.base.MultiTrace:
        return self.trace

    def _init_shared(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]):
        self.y = tt.shared(y) if y is not None else None
        self.X = {}
        self.size = {}
        for x in X:
            self.X[x] = tt.shared(X[x].values)
            self.size[x] = X[x].nunique() + 1  # +1 for non-obseved cases

        self.X_ = X.copy()

    def _set_shared(self, X):
        for x in self.X_:
            logger.debug(
                f"setting {x} to shared value (old shape {self.X_[x].shape}, new shape {X[x].shape})"
            )
            self.X[x].set_value(X[x].values)

    def _reset_shared(self):
        self._set_shared(self.X_)

    @contextmanager
    def _data_context(self, X: pd.DataFrame, *args, **kwargs):
        try:
            self._set_shared(X)
            yield None
        finally:
            self._reset_shared()

    def _init_model_context(self):
        self.model = pm.Model()

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self._init_shared(X, y)
        self._mean_trace = {}
        self._init_model_context()
        self.trace = self.model_block()
        return self

    def predict(self, X, mean=False, fast=True, **kwargs):
        with self._data_context(X):

            if fast:
                sample_ppc = pm.fast_sample_posterior_predictive
            else:
                sample_ppc = pm.sample_posterior_predictive

            ppc = sample_ppc(
                trace=self.trace,
                model=self.model,
                **kwargs
            )

            if len(ppc) > 1: # multi-response, deal with later
                logger.warning(
                    "multiple responses found in pymc3 model context "
                    "returning dict of arrays rather than arrays themselves."
                )
                return ppc
            
            k = list(ppc)[0]
            if mean:
                return ppc[k].mean(axis=0)
            return ppc[k]
