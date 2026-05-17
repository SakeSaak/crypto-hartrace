"""HAR-family density forecasting for BTC-EUR realized volatility.

Submodules
----------
distributions : Hansen (1994) skewed-t PDF/CDF/PPF/RVS/log-likelihood.
params        : Dataclasses for fitted model parameters.
state         : HARState representation of ℱ_t.
dynamics      : HAR mean equation + σ_t recursions (static/GAS/EGARCH).
simulator     : Multi-step Monte Carlo density forecaster.
"""

from .distributions import pdf, cdf, ppf, rvs, loglik  # noqa: F401
from .params import (  # noqa: F401
    ModelSpec, HARMeanParams, ScaleParams,
    InnovationParams, ReturnInnovationParams,
)
from .state import HARState  # noqa: F401
from .dynamics import har_mean, update_sigma  # noqa: F401
from .simulator import (  # noqa: F401
    simulate, SimulationResult, AuxForecasters, ComponentSampler,
)

__version__ = "0.0.1"
