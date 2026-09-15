import pandas as pd 
import numpy as np
import bindfit
import pprint

def run_bindfit(input_file: str, hostname: str, guestname: str, fitter_name: str, method: str, normalise: bool, flavour: str, dilute: bool, params: dict) -> dict:
    """Run bindfit on a given input file.

    Parameters
    ----------
    input_file : str
        Path to input file
    hostname : str
        Name of host column
    guestname : str
        Name of guest column
    fitter_name : str
        Name of fitter to use
    method : str
        Name of fitting method
    normalise : bool
        Normalise data
    dilute : bool
        Dilute data
    flavour : str
        Flavour of fitter
    params : dict
        Fitting parameters

    Returns
    -------
    dict
        Fitting summary
    """

    # Load raw data
    data = pd.read_csv(input_file)

    # Set index as per fitter
    data = data.set_index([hostname, guestname])

    function = bindfit.functions.construct(
        fitter_name,
        normalise=normalise,
        flavour=flavour,
    )

    fitter = bindfit.fitter.Fitter(
        data, function, normalise=normalise, params=params, dilution_correction=dilute
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

    return summary, fitter

def assertValueInRange(self, value, target, error):
    """Check that a value is within target +/- error.

    Parameters
    ----------
    self : unittest.TestCase
        Test case instance
    value : float
        Value to check
    target : float
        Target value
    error : float
        Error margin

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If value is not within target +/- error
    """
    self.assertTrue((value < target + error) and (value > target - error))