"""Weibull life analysis and spare-parts planning for repairable fleets."""

from .forecast import LeadTimeDemand, lead_time_demand, poisson_binomial_pmf, renewal_share, simulate_failures
from .stocking import cost_optimal_stock, optimal_replacement_age, stock_for_service_level, stock_options
from .weibull import Fit, Weibull, fit, fit_mle, fit_rank_regression, median_ranks

__all__ = [
    "Fit",
    "LeadTimeDemand",
    "Weibull",
    "cost_optimal_stock",
    "fit",
    "fit_mle",
    "fit_rank_regression",
    "lead_time_demand",
    "median_ranks",
    "optimal_replacement_age",
    "poisson_binomial_pmf",
    "renewal_share",
    "simulate_failures",
    "stock_for_service_level",
    "stock_options",
]

__version__ = "1.0.0"
