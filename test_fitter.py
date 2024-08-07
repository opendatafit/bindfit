#!/usr/bin/env python

import numpy as np
import bindfit
import pprint
import pandas as pd


# Parameters and options
params = {
    "k": {
        "init": 100.0,
        "bounds": {
            "min": 0.0,
            "max": None,
        },
    },
}

fitter_name = "nmr1to1"
method = "Nelder-Mead"
normalise = True
dilute = False
flavour = "none"

# Load raw data
data = pd.read_csv("input.csv")

# Set index as per fitter
data = data.set_index(["Host", "Guest"])

# Dilution correction is already in fitter
# if dilute:
#    data_y = bindfit.helpers.dilute(data_x[0], data_y)

function = bindfit.functions.construct(
    fitter_name,
    normalise=normalise,
    flavour=flavour,
)

fitter = bindfit.fitter.Fitter(
    data, function, normalise=normalise, params=params
)

fitter.run_scipy(params, method=method)

summary = {
    "fitter": fitter_name,
    "fit": {
        "y": fitter.fit,
        "coeffs_raw": fitter.coeffs_raw,
        "coeffs": fitter.coeffs,
        "molefrac_raw": fitter.molefrac_raw,
        "molefrac": fitter.molefrac,
        "params": fitter.params,
        "n_y": np.array(fitter.fit).size,
        "n_params": len(fitter.params) + np.array(fitter.coeffs_raw).size,
    },
    "qof": {
        "residuals": fitter.residuals,
        "ssr": bindfit.helpers.ssr(fitter.residuals),
        "rms": bindfit.helpers.rms(fitter.residuals),
        "cov": bindfit.helpers.cov(fitter.ydata, fitter.residuals),
        "rms_total": bindfit.helpers.rms(fitter.residuals, total=True),
        "cov_total": bindfit.helpers.cov(
            fitter.ydata, fitter.residuals, total=True
        ),
    },
    "time": fitter.time,
    "options": {
        "dilute": dilute,
        "normalise": normalise,
        "method": method,
        "flavour": flavour,
    },
}

pprint.pprint(summary)
