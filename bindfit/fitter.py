import time
from itertools import product

import pandas as pd
import numpy as np
import numpy.matlib as ml

import scipy
import scipy.optimize
from scipy import stats

from . import helpers


class Fitter:
    """Fitter class for optimising binding constant functions.

    Parameters
    ----------
    data : pandas.DataFrame, MxN matrix
        Input data matrix with M columns of index variables (e.g. Host, Guest)
        and N columns observed data variables
    function : function
        The fitter function to use for optimisation
    normalise : boolean, optional
        Whether to subtract initial x values before fitting, defaults to True
    params : dict
        Dict of initial values for parameters to pass to the fitting func
        Value of parameter keys depends on the fitting function selected
        Example:
        {
            "k": {
                "init": 100.0,
                "bounds": {
                    "min": 0.0,
                    "max": None,
                },
            },
        }

    Attributes
    ----------
    xdata : array_like, 2xN matrix
        Host/Guest data matrix, one variable per row
    ydata : array_like, MxN matrix
        Observed data matrix, one variable per row
    function : function
        The fitter function to use for optimisation
    normalise : boolean, optional
        Whether to normalise the x data before fitting
    dilution_correction: boolean, optional
        Whether to apply dilution correction to the y data before fitting
    params : dict
        Dict of initial values for parameters passed to the fitting func
        See above for example format
    time : string
        Populated after fitting, total time taken to fit
    fit : array_like, MxN matrix
        Fit curve matrix, same dimensions as ydata
    residuals : array_like, MxN matrix
        Observed data fit residuals matrix, same dimensions as ydata
    coeffs : array_like, (1:1 - 2|1:2 - 3)xM matrix
        Fit coefficients
    molefrac : array_like, (1:1 - 2|1:2 - 3)xN matrix
        Fit molefractions
    """

    # Dict mapping model function names to coefficient names
    MODEL_COEFFS_MAP = {
        "nmr_1to1": ["H", "HG"],
        "nmr_1to2": ["H", "HG", "HG2"],
        "nmr_2to1": ["H", "HG", "H2G"],
        "nmr_dimer": ["H", "Hs", "He"],
        "nmr_coek": ["H", "Hs", "He"],
        "uv_1to1": ["H", "HG"],
        "uv_1to2": ["H", "HG", "HG2"],
        "uv_1to3": ["H", "HG", "HG2", "HG3"],
        "uv_2to1": ["H", "HG", "H2G"],
        "uv_3to1": ["H", "HG", "H2G", "H3G"],
        "uv_dimer": ["H", "Hs", "He"],
        "uv_coek": ["H", "Hs", "He"],
    }

    def __init__(
        self,
        data,
        function,
        # Initial parameter guesses for the fit
        params=None,
        # Whether to subtract initial values before fitting
        normalise=True,
        # Whether to apply dilution correction before fitting
        dilution_correction=False,
    ):
        # Data in pandas DataFrame format, with x columns set as index
        self.data = data

        # Munge data into old expected format
        # TODO: Ideally should modify other functions to use dataframe format
        # But this is the quickest solution for the moment
        # Bindfit expects each variable as row
        # np.asarray(list()) required to convert index list of tuples to normal
        # numpy matrix
        # xdata : array_like, 2xN matrix
        #     Host/Guest data matrix, one variable per row
        # ydata : array_like, MxN matrix
        #     Observed data matrix, one variable per row
        self.xdata = np.transpose(np.asarray(list(data.index.to_numpy())))
        self.ydata = np.transpose(data.to_numpy())

        self.function = function

        # Fitter options
        self.normalise = normalise
        self.dilution_correction = dilution_correction

        # Apply dilution correction
        # TODO: Does this belong here? Or should it only be applied temporarily
        # during the fit?
        if dilution_correction:
            self.ydata = helpers.dilute(self.xdata[0], self.ydata)

        # Populated on Fitter.run
        self._params_raw = None
        self.params = params  # Initialise with optimised param results
        # from previous run
        # Used with post-fit Monte Carlo calculation
        self.time = None
        self.fit = None
        self.residuals = None
        self.coeffs = None
        self.molefrac = None

    def _preprocess(self, ydata):
        # Preprocess data based on Fitter options
        # Returns modified processed copy of input data
        d = ydata

        if self.normalise:
            d = helpers.normalise(d)

        return d

    def _postprocess(self, ydata, yfit):
        # Postprocess fitted data based on Fitter options
        f = yfit

        if self.normalise:
            f = helpers.denormalise(ydata, yfit)

        return f

    def run_scipy(
        self,
        params_init,
        save=True,
        xdata=None,
        ydata=None,
        method="Nelder-Mead",
    ):
        """Fit data given initial parameter guesses.

        Parameters
        ----------
        params_init : `dict`
            Initial parameter guesses for fitter.
        save : `boolean`
            If True, process and save optimisation results.
            If False, return raw optimised params.
        xdata : `ndarray`, optional
            Modified input array.
            (used with save=False for Monte Carlo error calculation)
        ydata : `ndarray`, optional
            Modified input array.
            (used with save=False for Monte Carlo error calculation)
        method : `string`, optional
            The fitting method to use.
        """
        # Set input data
        x = self.xdata if xdata is None else xdata
        y = self._preprocess(self.ydata if ydata is None else ydata)

        # Sort parameter dict into ordered array of parameters and bounds
        p = []
        b = []
        for key, value in sorted(params_init.items()):
            p.append(value["init"])
            b.append([value["bounds"]["min"], value["bounds"]["max"]])

        # Run optimizer
        tic = time.perf_counter()
        result = scipy.optimize.minimize(
            self.function.objective,
            p,
            bounds=b,
            args=(x, y, True),
            method=method if method else "Nelder-Mead",
            tol=1e-18,
        )
        toc = time.perf_counter()

        # Calculate fitted data with optimised parameters.
        # Force molefraction (not free concentration) calculation for proper
        # fitting in UV models.
        ydata_init = self.ydata[:, 0] if ydata is None else ydata[:, 0]
        (
            fit_norm,
            residuals,
            coeffs_raw,
            molefrac_raw,
            coeffs,
            molefrac,
        ) = self.function.objective(
            result.x, x, y, scalar=False, ydata_init=ydata_init
        )

        # Postprocessing
        # Populate fit results dict
        results = {}

        # Save time taken to fit
        results["time"] = toc - tic

        # Save raw optimised params arra
        results["_params_raw"] = result.x

        # Postprocess (denormalise) and save fitted data
        fit = self._postprocess(
            self.ydata if ydata is None else ydata, fit_norm
        )
        results["fit"] = fit

        results["residuals"] = residuals

        results["coeffs"] = coeffs
        results["coeffs_raw"] = coeffs_raw

        # Calculate final molefrac (including host, function-specific)
        results["molefrac"] = molefrac
        results["molefrac_raw"] = molefrac_raw

        # Calculate fit uncertainty statistics
        err = self.statistics(result.x, fit, coeffs_raw, residuals)

        # Parse final optimised parameters and errors into parameters dict
        results["params"] = self.function.format_params(
            params_init, result.x, err
        )

        if save:
            # Save fit results dict to object instance
            for key, value in results.items():
                setattr(self, key, value)

            # self.calc_monte_carlo(5, [0.02, 0.01], 0.005)
        else:
            # Return results dict without saving
            return results

    def statistics(self, params, fit, coeffs, residuals):
        """Calculate fit statistics.

        Returns
        -------
        ci : `float`
            Confidence interval.
            Asymptotic error for non-linear parameter estimate.
        # Standard deviation of calculated y
        # Standard deviation of calculated coefficients
        """
        # Calculate deLevie uncertainty
        d = np.float64(1e-6)  # delta

        # 0. Calculate partial differentials for each parameter
        diffs = []
        for i, pi in enumerate(params):
            # Shift the ith parameter's value by delta
            pi_shift = pi * (1 + d)
            params_shift = np.copy(params)
            params_shift[i] = pi_shift

            # Calculate fit with modified parameter set
            x = self.xdata
            y = self._preprocess(self.ydata)
            ydata_init = self.ydata[:, 0]
            fit_shift_norm, _, _, _, _, _ = self.function.objective(
                params_shift,
                x,
                y,
                scalar=False,
                ydata_init=ydata_init,
                fit_coeffs=coeffs,
            )
            fit_shift = self._postprocess(self.ydata, fit_shift_norm)

            # Calculate partial differential
            # Flatten numerator into 1D array (TODO: is this correct?)
            num = (fit_shift - fit).flatten()
            denom = pi_shift - pi
            diffs.append(np.divide(num, denom))

        diffs = np.array(diffs)

        # 1. Calculate PxP matrix M and invert
        P = len(params)
        M = np.zeros((P, P))
        for i, j in product(range(P), range(P)):
            M[i, j] = np.sum(diffs[i] * diffs[j])

        M_inv = np.linalg.inv(M)
        m_diag = np.diagonal(M_inv)

        # 2. Calculate standard deviations sigma of P parameters pi
        # Sum of squares of residuals
        ssr = np.sum(np.square(residuals))
        # Degrees of freedom:
        # N datapoints - N fitted params - N calculated coefficients
        d_free = self.ydata.size - len(params) - coeffs.size

        sigma = np.sqrt((m_diag * ssr) / (d_free - 1))

        # 3. Calculate confidence intervals
        # Calculate t-value at 95%
        # Studnt, n=d_free, p<0.05, 2-tail
        t = stats.t.ppf(1 - 0.025, d_free)

        # ci = np.array([params - t * sigma, params + t * sigma])
        ci_percent = (t * sigma) / params * 100

        return ci_percent

    def calc_monte_carlo(self, n_iter, xdata_error, ydata_error, method=None):
        """Calculate fit error using Monte Carlo method.

        Parameters
        ----------
        n_iter : `int`
            Number of Monte Carlo iterations.
        xdata_error : `ndarray`
            N array of N percentage errors corresponding to N rows of xdata.
        ydata_error : `float`
            Float corresponding to N percentage error on each row of ydata.

        Returns
        -------
        Something.
        """
        xdata = self.xdata
        ydata = self.ydata

        # Convert params results to params_init array to use as input
        # to run_scipy
        # TODO: make params_init same format as params result
        # params_init = {}
        # for key, param in self.params.items():
        #     params_init[key] = param["value"]

        # Copy parameter results array and set inital values to optimised
        # parameter results to use as input to run_scipy
        params_init = {}
        for key, param in self.params.items():
            params_init[key] = param
            params_init[key]["init"] = param["value"]

        params_arr = np.zeros((n_iter, len(params_init)))

        for n in range(n_iter):
            # Calculate error multiplier arrays matching ydata, xdata shapes
            xdata_error_arr = (
                np.random.standard_normal(xdata.shape)
                * ml.repmat(xdata_error, xdata.shape[1], 1).T
                + 1
            )
            ydata_error_arr = (
                np.random.standard_normal(ydata.shape) * ydata_error + 1
            )

            # Calculated shifted input data
            xdata_shift = xdata * xdata_error_arr
            ydata_shift = ydata * ydata_error_arr

            results = self.run_scipy(
                params_init=params_init,
                save=False,
                xdata=xdata_shift,
                ydata=ydata_shift,
                method=method,
            )

            # Log resulting params
            params_arr[n] = results["_params_raw"]

        percentile_params = np.percentile(params_arr, [2.5, 97.5], axis=0).T

        # Calculate errors and update input params dict with results
        for i, (key, param) in enumerate(sorted(self.params.items())):
            p = param["value"]  # Actual param result
            per = percentile_params[i]  # Calc'd percentile for this param

            lower = (100 * (per[0] - p)) / p
            upper = (100 * (per[1] - p)) / p

            param["mc"] = [lower, upper]

        return self.params

    @property
    def fit_curve(self):
        """Return fit curve data as pandas DataFrame"""
        fit_curve = self.data.copy(deep=True)
        fit_curve[:] = np.transpose(self.fit)
        return fit_curve

    @property
    def fit_residuals(self):
        """Return fit residuals data as pandas DataFrame"""
        fit_residuals = self.data.copy(deep=True)
        fit_residuals[:] = np.transpose(self.fit - self.ydata)
        return fit_residuals

    @property
    def fit_molefractions(self):
        """Return optimised molefractions table as pandas DataFrame"""
        # Build DataFrame of molefractions with x vars and column names
        return (
            pd.DataFrame(np.transpose(self.molefrac))
            .set_index(self.data.index)
            .set_axis(self.MODEL_COEFFS_MAP[self.function.f.__name__], axis=1)
        )

    @property
    def fit_coefficients(self):
        """Return optimised coefficients table as pandas DataFrame"""
        return (
            pd.DataFrame(np.transpose(self.coeffs))
            # Set index to fit column names
            .set_index(self.data.columns.rename("name"))
            # Set coefficient column names
            .set_axis(self.MODEL_COEFFS_MAP[self.function.f.__name__], axis=1)
        )

    @property
    def fit_summary(self):
        """Return fit summary data as pandas DataFrame"""
        return pd.DataFrame(
            [
                [
                    self.function.f.__name__,
                    self.time,
                    helpers.ssr(self.residuals),
                    np.array(self.fit).size,
                    len(self.params) + np.array(self.coeffs_raw).size,
                ]
            ],
            columns=[
                "Model",
                "Time",
                "SSR",
                "Fitted datapoints",
                "Fitted parameters",
            ],
        ).set_index("Model")

    @property
    def fit_quality(self):
        """Return fit quality statistics as pandas DataFrame"""
        return (
            pd.DataFrame(
                np.transpose(
                    [
                        helpers.rms(self.residuals),
                        helpers.cov(self.ydata, self.residuals),
                    ]
                ),
                columns=["RMS", "Covariance"],
            )
            # Set index to fit column names
            # Doing it this way instead of setting index in constructor as
            # this allows us to use a custom named index
            .set_index(self.data.columns.rename("Fit"))
        )
