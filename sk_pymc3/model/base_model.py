import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import theano as tt

from collections.abc import ABC, abstractmethod
from contextlib import contextmanager
from sklearn.base import BaseEstimator
from typing import List, Optional, Union

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
            self.X[x] = tt.shared(X[x])
            self.size[x] = X[x].nunique() + 1  # +1 for non-obseved cases

        self.X_ = X.copy()

    def _set_shared(self, X)
        for x in self.X_:
            self.X[x].set_value(X[x])

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

    def _construct_mean_trace(self, force=False):
        if self._mean_trace and not force:
            return

        for k in self.trace.varnames:
            self._mean_trace[k] = self.trace[k].mean(axis=0)

        self._mean_trace = [self._mean_trace]

    def predict(self, X, mean=False, **kwargs):
        with self._data_context(X):

            trace = self.trace if not mean else self._mean_trace
            ppc = pm.sample_posterior_predictive(
                trace=trace,
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
            return ppc[k]