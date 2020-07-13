import logging
import numpy as np
import pandas as pd
import pymc3 as pm

from pymc3 import get_data
from sk_pymc3.model import BasePyMC3Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_close(x, y, eps):
    return np.logical_or(x <= y + eps / 2, x >= y - eps / 2)


def test_base_model_on_radon_data():
    
    srrs2 = pd.read_csv(get_data('srrs2.dat'))
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state=='MN'].copy()
    srrs_mn['fips'] = srrs_mn.stfips*1000 + srrs_mn.cntyfips
    cty = pd.read_csv(get_data('cty.dat'))
    cty_mn = cty[cty.st=='MN'].copy()
    cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

    srrs_mn["log_radon"] = np.log(
        srrs_mn["activity"] + 0.1
    )

    class PooledModel(BasePyMC3Model):
        
        def model_block(self):
            
            with self.model:
                
                sd = pm.HalfCauchy("sigma", 5)
                beta_floor = pm.Normal("beta_floor", 0, 100, shape=2)
                theta = beta_floor[0] + beta_floor[1] * self.X["floor"]
                
                y = pm.Normal("y", theta, sd, observed=self.y)
                
                return pm.sample(random_seed=42069)
    
    logger.info("instantiating model")
    pooled_model = PooledModel()
    logger.info("fitting")
    pooled_model = pooled_model.fit(srrs_mn, srrs_mn["log_radon"].values)
    logger.info("fitted")
    betas = pooled_model.trace["beta_floor"].mean(axis=0)
    assert is_close(betas, np.array([1.36, -.587]), .05).all()
   
    ppc = pooled_model.predict(srrs_mn.head(1), mean=True)
    assert ppc.shape[0] == 1

    ppc = pooled_model.predict(srrs_mn.head(1), mean=False)
    assert ppc.shape[-1] == 1

    ppc = pooled_model.predict(srrs_mn.head(10), mean=True)
    assert ppc.shape[0] == 10

    ppc = pooled_model.predict(srrs_mn.head(10), mean=False)
    assert ppc.shape[-1] == 10
