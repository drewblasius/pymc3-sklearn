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
    
    @staticmethod
    def _rep_frame_if_singleton(X: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        if X.shape[0] > 1:
            return X, False
        return pd.concat([X] * 2), True
    
    @staticmethod
    def _post_unrep_ppc(ppc_dict: dict):
        dk = list(ppc_dict)
        for k in dk:
            ppc_dict[k] = ppc_dict[k][..., :1]
        return ppc_dict

    def predict(self, X, mean=False, fast=True, **kwargs):
        X, rep = self._rep_frame_if_singleton(X)

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
            
            # Theano broadcasts shared things incorrectly if singletons.
            if rep:
                ppc = self._post_unrep_ppc(ppc)

            if len(ppc) > 1:  # multi-response, deal with later
                logger.warning(
                    "multiple responses found in pymc3 model context "
                    "returning dict of arrays rather than arrays themselves."
                )
                return ppc
            
            k = list(ppc)[0]
            if mean:
                return ppc[k].mean(axis=0)
            return ppc[k]
