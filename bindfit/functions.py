"""Binding constant minimisation function classes."""


import numpy as np


class BaseFunction:
    """Base Function abstract class.

    To use, choose an objective function and plotting mixin and create a class
    like this:

    `class Function(PlotMixin, ObjectiveMixin, BaseFunction)`

    Note all mixins are placed before BaseFunction. This is necessary for the
    mixin functions to override BaseFunction template functions.

    See here: https://www.ianlewis.org/en/mixins-and-python

    Attributes
    ----------
    f : `function`
        Fitting function, defined below e.g. nmr_1to1(...).
    fitter : `string`
        Fitting model name. Example: `nmr1to1`
    normalise : `boolean`
        If true, subtract initial values from Y data.
    flavour : `string`
        Fitting function flavour.
        One of: `none`, `add`, `stat`, `noncoop`.
    """

    def __init__(self, fitter, f=None, normalise=True, flavour="none"):
        self.f = f
        self.fitter = fitter
        self.normalise = normalise
        self.flavour = flavour

    def objective(
        self,
        params,
        xdata,
        ydata,
        scalar=False,
        ydata_init=None,
        fit_coeffs=None,
        *args,
        **kwargs,
    ):
        """Objective function definition.

        Parameters
        ----------
        params : `dict`
            Initial parameter value guesses.
        datax : `ndarray`
            X x M array of X independent variables, M observations.
        datay : `ndarray`
            Y x M array of Y dependent variables, M observations.
        scalar: `bool`
            If true, calculate and return only the sum of least squares (SSR).
        ydata_init : `ndarray`, required if `scalar=False`
            Array of initial Y data values, length Y, required if scalar=False.
        fit_coeffs : `ndarray`, optional
            If provided, use pre-calculated coefficient values.
            Used for error calculations.

        Returns
        -------
        ssr : float
            Sum of least squares

        OR

        fit : `ndarray`
            Y fit data, same dimensions as datay.
        residuals : `ndarray`
            Y residuals, same dimensions as datay.
        coeffs_raw : `ndarray`
            (1:1 - 1|1:2 - 2) x Y matrix of raw fit coefficients.
        molefrac_raw : `ndarray`
            (1:1 - 1|1:2 - 2) x Y matrix of raw molefractions.
        coeffs : `ndarray`
            Fit coefficients with first row of initial values added.
        molefrac : `ndarray`
            Molefractions with first row of initial values added.
        """
        pass

    def format_x(self, xdata):
        pass

    def format_coeffs(self, coeffs, ydata_init, h0_init=None):
        """Calculate "real" coefficients from their raw values and an input
        dataset.

        Calculate and add first row of coefficients using excluded initial
        values.

        Parameters
        ----------
        ydata_init : `ndarray`
            1 x M array of non-normalised initial observations of dependent
            variables
        coeffs : `ndarray`
            (1:1 - 1|1:2 - 2) x Y matrix of raw fit coefficients.
        ydata_init : `ndarray`
            Initial input Y data.
            Y x M array of Y dependent variables, M observations.
        h0_init : float
            Optional initial h0 value, if provided ydata_init is divided by
            this value before the calculation

        Returns
        -------
        coeffs : `ndarray`
            (1:1 - 2|1:2 - 3) x Y matrix of processed fit coefficients.
        """
        pass

    def format_params(self, params_init, params_raw, err):
        pass


# =============================================================================
# Objective function mixins


class BindingMixin:
    def objective(
        self,
        params,
        xdata,
        ydata,
        scalar=True,
        ydata_init=None,
        fit_coeffs=None,
        *args,
        **kwargs,
    ):
        """Binding constant objective function.

        Performs least squares regression fitting via matrix division on
        provided NMR/UV dataset for a given binding constant K, and returns its
        sum of least squares for optimisation OR full parameters, residuals and
        fitted results.
        """
        # Calculate predicted HG complex concentrations for this set of
        # parameters and concentrations
        molefrac_raw, molefrac = self.f(params, xdata, flavour=self.flavour)

        if self.normalise:
            # Don't fit first H column if initial values subtracted
            molefrac_raw = molefrac_raw[1:]

        if fit_coeffs is not None:
            coeffs_raw = fit_coeffs
        else:
            # Solve by matrix division - linear regression by least squares
            # This is equivalent to
            # << coeffs = molefrac\ydata (EA = HG\DA) >>
            # in Matlab
            coeffs_raw, _, _, _ = np.linalg.lstsq(
                molefrac_raw.T, ydata.T, rcond=-1
            )

        # Restrict UV coefficients to positive values when normalised
        if not self.normalise and "uv" in self.fitter:
            coeffs_raw[coeffs_raw < 0] = 0

        # Calculate data from fitted parameters
        # (will be normalised if input data was normalised)
        # Result is column matrix - transpose this into same shape as input
        # data array
        fit = molefrac_raw.T.dot(coeffs_raw).T

        # Calculate residuals (fitted data - input data)
        residuals = fit - ydata

        if scalar:
            return np.square(residuals).sum()
        else:
            # Return full fit with formatted molefrac and coeffs
            coeffs = self.format_coeffs(
                coeffs_raw, ydata_init=ydata_init, h0_init=xdata[0][0]
            )

            return fit, residuals, coeffs_raw, molefrac_raw, coeffs, molefrac

    def format_x(self, xdata):
        h0 = xdata[0]
        g0 = xdata[1]
        return g0 / h0

    def format_coeffs(self, coeffs, ydata_init, h0_init=None):
        # H coefficients
        h = np.copy(ydata_init)
        coeffs = np.array(coeffs)

        if self.flavour == "add" or self.flavour == "stat":
            # Preprocess coeffs for additive flavours
            # Calculate HG2/H2G coeffs and add to returned HG coeffs array
            if self.normalise:
                coeffs = np.array([coeffs[0], coeffs[0] * 2])
            else:
                # Account for first row of H
                coeffs = np.array([coeffs[0], coeffs[1], coeffs[1] * 2])

        if self.normalise:
            if "uv" in self.fitter and h0_init is not None:
                # Calculate coefficient for H (ydata_init/h0)
                h /= h0_init

            # Calc and add first row of coeffs using excluded initial values
            rows = coeffs.shape[0]
            if rows == 1:
                # 1:1 system
                hg = h + coeffs[0]
                return np.vstack((h, hg))
            elif rows == 2:
                # 1:2 or 2:1 system
                hg = h + coeffs[0]
                hg2 = h + coeffs[1]
                return np.vstack((h, hg, hg2))
            else:
                pass  # Throw error here
        else:
            return coeffs

    def format_params(self, params_init, params_result, err):
        params = params_init

        for name, param, stderr in zip(
            sorted(params_init), params_result, err
        ):
            params[name].update(
                {
                    "value": param,
                    "stderr": stderr,
                }
            )

        return params


class AggMixin:
    def objective(
        self,
        params,
        xdata,
        ydata,
        scalar=False,
        ydata_init=None,
        fit_coeffs=None,
        *args,
        **kwargs,
    ):
        """Dimer aggregation objective function."""
        # Calculate predicted complex concentrations for this set of
        # parameters and concentrations
        molefrac_raw, molefrac = self.f(params, xdata, flavour=self.flavour)
        h = molefrac_raw[0]
        hs = molefrac_raw[1]
        he = molefrac_raw[2]
        hmat = np.array([h + he / 2, hs + he / 2])

        # Solve by matrix division - linear regression by least squares
        # Equivalent to << coeffs = molefrac\ydata (EA = HG\DA) >> in Matlab
        if fit_coeffs is not None:
            coeffs_raw = fit_coeffs
        else:
            coeffs_raw, _, _, _ = np.linalg.lstsq(hmat.T, ydata.T)

        # Calculate data from fitted parameters
        # (will be normalised since input data was norm'd)
        # Result is column matrix - transform this into same shape as input
        # data array
        fit = hmat.T.dot(coeffs_raw).T

        # Calculate residuals (fitted data - input data)
        residuals = fit - ydata

        # Transpose any column-matrices to rows
        if scalar:
            return np.square(residuals).sum()
        else:
            # Return full fit with formatted molefrac and coeffs
            coeffs = self.format_coeffs(
                coeffs_raw, ydata_init=ydata_init, h0_init=xdata[0][0]
            )
            return fit, residuals, coeffs_raw, hmat, coeffs, molefrac

    def format_x(self, xdata):
        return xdata[0]

    def format_coeffs(self, coeffs, ydata_init, h0_init=None):
        # H coefficients
        coeffs = np.array(coeffs)

        # TODO: Proper coefficient formatting here

        return coeffs

    def format_params(self, params_init, params_result, err):
        params = params_init

        for name, param, stderr in zip(
            sorted(params_init), params_result, err
        ):
            if name == "ke":
                params[name].update(
                    {
                        "value": [param, param / 2],  # Calculate Kd if Ke
                        "stderr": [stderr, stderr / 2],  # parameter given
                    }
                )
            else:
                params[name].update(
                    {
                        "value": param,  # Otherwise single param value
                        "stderr": stderr,
                    }
                )

        return params


# =============================================================================
# Final class definitions


class FunctionBinding(BindingMixin, BaseFunction):
    pass


class FunctionAgg(AggMixin, BaseFunction):
    pass


# =============================================================================
# log(inhibitor) vs. normalised response test definition


class FunctionInhibitorResponse(FunctionBinding):
    """log(inhibitor) vs. normalised response test definition."""

    def objective(self, params, xdata, ydata, scalar=False, *args, **kwargs):
        yfit = self.f(params, xdata)
        yfit = yfit[np.newaxis]

        # Calculate residuals (fitted data - input data)
        residuals = yfit - ydata

        if scalar:
            return np.square(residuals).sum()
        else:
            return (
                yfit,
                residuals,
                np.zeros(1, dtype="float64"),
                np.zeros((1, 1), dtype="float64"),
            )


def inhibitor_response(params, xdata, *args, **kwargs):
    """Calculates predicted [HG] given data object parameters as input."""
    # Params sorted in alphabetical order
    hillslope = params[0]
    logIC50 = params[1]

    inhibitor = xdata[1]  # xdata[0] is just 1s to fudge geq calc

    response = 100 / (1 + 10 ** ((logIC50 - inhibitor) * hillslope))

    return response


# =============================================================================
# Fitting function definitions


def nmr_1to1(params, xdata, *args, **kwargs):
    """Calculates predicted [HG] given data object parameters as input."""
    k = params[0]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices
    # and Ka guess
    hg = 0.5 * (
        (g0 + h0 + (1 / k))
        - np.lib.scimath.sqrt(((g0 + h0 + (1 / k)) ** 2) - (4 * ((g0 * h0))))
    )
    h = h0 - hg

    # Replace any non-real solutions with sqrt(h0*g0)
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Convert [HG] concentration to molefraction for NMR
    hg /= h0
    h /= h0

    # Make column vector
    # hg_mat = hg[np.newaxis]
    hg_mat_fit = np.vstack((h, hg))
    hg_mat = np.vstack((h, hg))

    return hg_mat_fit, hg_mat


def uv_1to1(params, xdata, *args, **kwargs):
    """Calculates predicted [HG] given data object parameters as input."""
    k = params[0]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices
    # and Ka guess
    hg = 0.5 * (
        (g0 + h0 + (1 / k))
        - np.lib.scimath.sqrt(((g0 + h0 + (1 / k)) ** 2) - (4 * ((g0 * h0))))
    )
    h = h0 - hg

    # Replace any non-real solutions with sqrt(h0*g0)
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Make column vector
    hg_mat_fit = np.vstack((h, hg))  # Free concentration for correct fitting
    hg_mat = np.vstack((h / h0, hg / h0))  # Molefrac for display

    return hg_mat_fit, hg_mat


def uv_1to2(params, xdata, flavour="none", *args, **kwargs):
    """Calculates predicted [HG] and [HG2] given data object and binding
    constants as input.
    """

    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11 / 4
    else:
        k12 = params[1]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate free guest concentration [G]: solve cubic
    a = np.ones(h0.shape[0]) * k11 * k12
    b = 2 * k11 * k12 * h0 + k11 - g0 * k11 * k12
    c = 1 + k11 * h0 - k11 * g0
    d = -1.0 * g0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [G] for each observation
    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        g[i] = soln

    # Calculate [HG] and [HG2] complex concentrations
    hg = h0 * ((g * k11) / (1 + (g * k11) + (g * g * k11 * k12)))
    hg2 = h0 * (((g * g * k11 * k12)) / (1 + (g * k11) + (g * g * k11 * k12)))
    h = h0 - hg - hg2

    if flavour == "add" or flavour == "stat":
        hg_add = hg + 2 * hg2
        hg_mat_fit = np.vstack((h, hg_add))
    else:
        hg_mat_fit = np.vstack((h, hg, hg2))

    hg_mat = np.vstack((h / h0, hg / h0, hg2 / h0))  # Display-only molefracs
    return hg_mat_fit, hg_mat


def nmr_1to2(params, xdata, flavour="none", *args, **kwargs):
    """Calculates predicted [HG] and [HG2] given data object and binding
    constants as input.
    """

    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11 / 4
    else:
        k12 = params[1]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate free guest concentration [G]: solve cubic
    a = np.ones(h0.shape[0]) * k11 * k12
    b = 2 * k11 * k12 * h0 + k11 - g0 * k11 * k12
    c = 1 + k11 * h0 - k11 * g0
    d = -1.0 * g0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [G] for each observation
    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        g[i] = soln

    # Calculate [HG] and [HG2] complex concentrations
    hg = (g * k11) / (1 + (g * k11) + (g * g * k11 * k12))
    hg2 = ((g * g * k11 * k12)) / (1 + (g * k11) + (g * g * k11 * k12))
    h = 1 - hg - hg2

    if flavour == "add" or flavour == "stat":
        hg_add = hg + 2 * hg2
        hg_mat_fit = np.vstack((h, hg_add))
    else:
        hg_mat_fit = np.vstack((h, hg, hg2))

    hg_mat = np.vstack((h, hg, hg2))
    return hg_mat_fit, hg_mat


def nmr_1to3(params, xdata, flavour="none", *args, **kwargs):
    """Calculates predicted [HG], [HG2], and [HG3] given data object and
    binding constants as input.
    """

    # Intialise Data
    k11 = params[0]
    if flavour == "noncoop":
        k12 = k11 / 3
        k13 = k11 / 9
    else:
        k12 = params[1]
        k13 = params[2]

    h0 = xdata[0]  # h0 in matlab code
    g0 = xdata[1]  # g0 in matlab code

    # Calculation of guest: Solve quartic
    a = np.ones(h0.shape[0]) * k11 * k12 * k13
    b = (k11 * k12) - (g0 * k11 * k12 * k13) + (3 * h0 * k11 * k12 * k13)
    c = k11 - (g0 * k11 * k12) + (2 * h0 * k11 * k12)
    d = 1 - (g0 * k11) + (h0 * k11)
    e = -1.0 * g0

    poly = np.column_stack((a, b, c, d, e))

    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        g[i] = soln

    hg = (g * k11) / (
        1 + (g * k11) + (g * g * k11 * k12) + (g * g * g * k11 * k12 * k13)
    )
    hg2 = (g * g * k11 * k12) / (
        1 + (g * k11) + (g * g * k11 * k12) + (g * g * g * k11 * k12 * k13)
    )
    hg3 = (g * g * g * k11 * k12 * k13) / (
        1 + (g * k11) + (g * g * k11 * k12) + (b * g * g * k11 * k12 * k13)
    )

    # h0 in UV
    h = 1 - hg - hg2 - hg3

    hg_mat_fit = np.vstack((h, hg, hg2, hg3))
    hg_mat = np.vstack((h, hg, hg2, hg3))

    return hg_mat_fit, hg_mat


def nmr_2to1(params, xdata, flavour="none", *args, **kwargs):
    """Calculates predicted [HG] and [H2G] given data object and binding
    constants as input.
    """
    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11 / 4
    else:
        k12 = params[1]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate free host concentration [H]: solve cubic
    a = np.ones(h0.shape[0]) * k11 * k12
    b = 2 * k11 * k12 * g0 + k11 - h0 * k11 * k12
    c = 1 + k11 * g0 - k11 * h0
    d = -1.0 * h0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])

    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # Calculate [HG] and [H2G] complex concentrations
    hg = (g0 * h * k11) / (h0 * (1 + (h * k11) + (h * h * k11 * k12)))
    h2g = (2 * g0 * h * h * k11 * k12) / (
        h0 * (1 + (h * k11) + (h * h * k11 * k12))
    )
    h = 1 - hg - h2g

    if flavour == "add" or flavour == "stat":
        hg_add = hg + 2 * h2g
        hg_mat_fit = np.vstack((h, hg_add))
    else:
        hg_mat_fit = np.vstack((h, hg, h2g))

    hg_mat = np.vstack((h, hg, h2g))
    return hg_mat_fit, hg_mat


def nmr_3to1(params, xdata, flavour="none", *args, **kwargs):
    """Calculates predicted [HG], [H2G], and [H3G] given data object and
    binding constants as input.
    """

    # Intialise Data
    k11 = params[0]
    if flavour == "noncoop":
        k12 = k11 / 3
        k13 = k11 / 9
    else:
        k12 = params[1]
        k13 = params[2]

    h0 = xdata[0]  # h0 in matlab code
    g0 = xdata[1]  # g0 in matlab code

    # Calculation of host: Solve quartic
    a = np.ones(h0.shape[0]) * k11 * k12 * k13
    b = (k11 * k12) - (g0 * k11 * k12 * k13) + (3 * h0 * k11 * k12 * k13)
    c = k11 - (g0 * k11 * k12) + (2 * h0 * k11 * k12)
    d = 1 - (g0 * k11) + (h0 * k11)
    e = -1.0 * g0

    poly = np.column_stack((a, b, c, d, e))

    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        g[i] = soln

    hg = (
        (1 / h0)
        * (g * k11)
        / (1 + (g * k11) + (g * g * k11 * k12) + (g * g * g * k11 * k12 * k13))
    )
    h2g = (
        (1 / h0)
        * (g * g * k11 * k12)
        / (1 + (g * k11) + (g * g * k11 * k12) + (g * g * g * k11 * k12 * k13))
    )
    h3g = (
        (1 / h0)
        * (g * g * g * k11 * k12 * k13)
        / (1 + (g * k11) + (g * g * k11 * k12) + (b * g * g * k11 * k12 * k13))
    )

    # We don't use h0 because NMR is chemical shift, UV is absorbance
    h = 1 - hg - h2g - h3g

    hg_mat_fit = np.vstack((h, hg, h2g, h3g))
    hg_mat = np.vstack((h, hg, h2g, h3g))

    return hg_mat_fit, hg_mat


def uv_2to1(params, xdata, flavour="none"):
    """Calculates predicted [HG] and [H2G] given data object and binding
    constants as input.
    """
    # Convenience
    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11 / 4
    else:
        k12 = params[1]

    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate free host concentration [H]: solve cubic
    a = np.ones(h0.shape[0]) * k11 * k12
    b = 2 * k11 * k12 * g0 + k11 - h0 * k11 * k12
    c = 1 + k11 * g0 - k11 * h0
    d = -1.0 * h0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])

    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # Calculate [HG] and [H2G] complex concentrations
    hg = g0 * ((h * k11) / (1 + (h * k11) + (h * h * k11 * k12)))
    h2g = g0 * (
        (2 * h * h * k11 * k12) / (1 + (h * k11) + (h * h * k11 * k12))
    )
    h = h0 - hg - h2g

    if flavour == "add" or flavour == "stat":
        hg_add = hg + 2 * h2g
        hg_mat_fit = np.vstack((h, hg_add))
    else:
        hg_mat_fit = np.vstack((h, hg, h2g))

    hg_mat = np.vstack((h / h0, hg / h0, h2g / h0))  # Molefrac for display
    return hg_mat_fit, hg_mat


def nmr_dimer(params, xdata, *args, **kwargs):
    """Calculates predicted [H] [Hs] and [He] given data object and binding
    constant as input.
    """
    ke = params[0]
    h0 = xdata[0]

    if ke == 0:
        # Avoid dividing by zero ...
        mf = np.array([h0 * 0, h0 * 0, h0 * 0])
        return mf, mf

    # Calculate free monomer concentration [H] or alpha:
    # eq 143 from Thordarson book chapter
    h = ((2 * ke * h0 + 1) - np.lib.scimath.sqrt(((4 * ke * h0 + 1)))) / (
        2 * ke * ke * h0 * h0
    )

    # Calculate "in stack" concentration [Hs] or epislon: eq 149
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs = (h * ((h * ke * h0) ** 2)) / ((1 - h * ke * h0) ** 2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1)
    # from Thordarson book chapter
    he = (2 * h * h * ke * h0) / (1 - h * ke * h0)

    mf_fit = np.vstack((h, hs, he))
    mf = np.vstack((h, hs, he))
    return mf_fit, mf


def uv_dimer(params, xdata, *args, **kwargs):
    """Calculates predicted [H] [Hs] and [He] given data object and binding
    constant as input.
    """
    ke = params[0]
    h0 = xdata[0]

    if ke == 0:
        # Avoid dividing by zero ...
        mf = np.array([h0 * 0, h0 * 0, h0 * 0])
        return mf, mf

    # Calculate free monomer concentration [H] or alpha:
    # eq 143 from Thordarson book chapter
    h = ((2 * ke * h0 + 1) - np.lib.scimath.sqrt(((4 * ke * h0 + 1)))) / (
        2 * ke * ke * h0 * h0
    )

    # Calculate "in stack" concentration [Hs] or epislon: eq 149
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs = (h0 * h * ((h * ke * h0) ** 2)) / ((1 - h * ke * h0) ** 2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1)
    # from Thordarson book chapter
    he = (h0 * (2 * h * h * ke * h0)) / (1 - h * ke * h0)

    # Convert to free concentration
    hc = h0 * h

    mf_fit = np.vstack((hc, hs, he))  # Free concentration for fitting
    mf = np.vstack((hc / h0, hs / h0, he / h0))  # Real molefraction
    return mf_fit, mf


def nmr_coek(params, xdata, *args, **kwargs):
    """Calculates predicted [H] [Hs] and [He] given data object and binding
    constants as input.
    """
    ke = params[0]
    rho = params[1]

    h0 = xdata[0]

    # Calculate free monomer concentration [H] or alpha:
    # eq 146 from Thordarson book chapter

    a = np.ones(h0.shape[0]) * (((ke * h0) ** 2) - (rho * ((ke * h0) ** 2)))
    b = 2 * rho * ke * h0 - 2 * ke * h0 - ((ke * h0) ** 2)
    c = 2 * ke * h0 + 1
    d = -1.0 * np.ones(h0.shape[0])

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # Calculate "in stack" concentration [Hs] or epislon:
    # eq 149 from Thordarson book chapter
    hs = (rho * h * ((h * ke * h0) ** 2)) / ((1 - h * ke * h0) ** 2)

    # Calculate "at end" concentration [He] or gamma:
    # eq 150 from Thordarson book chapter
    he = (2 * rho * h * h * ke * h0) / (1 - h * ke * h0)

    mf_fit = np.vstack((h, hs, he))
    mf = np.vstack((h, hs, he))
    return mf_fit, mf


def uv_coek(params, xdata, *args, **kwargs):
    """Calculates predicted [H] [Hs] and [He] given data object and binding
    constants as input.
    """
    ke = params[0]
    rho = params[1]

    h0 = xdata[0]

    # Calculate free monomer concentration [H] or alpha:
    # eq 146 from Thordarson book chapter

    a = np.ones(h0.shape[0]) * (((ke * h0) ** 2) - (rho * ((ke * h0) ** 2)))
    b = 2 * rho * ke * h0 - 2 * ke * h0 - ((ke * h0) ** 2)
    c = 2 * ke * h0 + 1
    d = -1.0 * np.ones(h0.shape[0])

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # n.b. these fractions are multiplied by h0

    # Calculate "in stack" concentration [Hs]
    # or epislon: eq 149 from Thordarson book chapter
    hs = (h0 * rho * h * ((h * ke * h0) ** 2)) / ((1 - h * ke * h0) ** 2)

    # Calculate "at end" concentration [He]
    # or gamma: eq 150 from Thordarson book chapter
    he = (h0 * (2 * rho * h * h * ke * h0)) / (1 - h * ke * h0)

    # Convert to free concentration
    hc = h0 * h

    mf_fit = np.vstack((hc, hs, he))  # Free concentration for fitting
    mf = np.vstack((hc / h0, hs / h0, he / h0))  # Real molefraction
    return mf_fit, mf


# =============================================================================
# Function class constructor helper


def construct(key, normalise=True, flavour="none"):
    """Constructs and returns the requested function object.

    Parameters
    ----------
    key : `string`
        Unique fitter function reference string, exposed by
        formatter.fitter_list
    flavour : `string`
        Fitting function flavour, if selected.
        One of: `none`, `add`, `stat`, `noncoop`.
    """

    args_select = {
        "nmrdata": ["FunctionBinding", (key)],
        "nmr1to1": ["FunctionBinding", (key, nmr_1to1, normalise, flavour)],
        "nmr1to2": ["FunctionBinding", (key, nmr_1to2, normalise, flavour)],
        "nmr1to3": ["FunctionBinding", (key, nmr_1to3, normalise, flavour)],
        "nmr2to1": ["FunctionBinding", (key, nmr_2to1, normalise, flavour)],
        "nmr3to1": ["FunctionBinding", (key, nmr_3to1, normalise, flavour)],
        "uvdata": ["FunctionBinding", (key)],
        "uv1to1": ["FunctionBinding", (key, uv_1to1, normalise, flavour)],
        "uv1to2": ["FunctionBinding", (key, uv_1to2, normalise, flavour)],
        "uv2to1": ["FunctionBinding", (key, uv_2to1, normalise, flavour)],
        "nmrdimer": ["FunctionAgg", (key, nmr_dimer, normalise, flavour)],
        "uvdimer": ["FunctionAgg", (key, uv_dimer, normalise, flavour)],
        "nmrcoek": ["FunctionAgg", (key, nmr_coek, normalise, flavour)],
        "uvcoek": ["FunctionAgg", (key, uv_coek, normalise, flavour)],
        "inhibitor": ["FunctionInhibitorResponse", (key, inhibitor_response)],
    }

    # Get appropriate class from global scope
    cls = globals()[args_select[key][0]]

    # Instantiation arguments
    args = args_select[key][1]

    # Construct and return
    return cls(*args)
