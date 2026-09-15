import unittest
import test_helpers
import numpy as np
import bindfit.functions as fc
import matplotlib.pyplot as plt

class TestBindfit(unittest.TestCase):
    # Test nmr1to1 fitter with Nelder-Mead method
    def test_nmr_1to1(self):
        input_file = r"tests/nmr1to1/NMR1to1.csv"
        hostname = "Host"
        guestname = "Guest"
        fitter_name = "nmr1to1"
        method = "Nelder-Mead"
        normalise = True
        dilute = False
        flavour = "none"
        params = {
        "k": {
                "init": 100.0,
                "bounds": {
                    "min": 0.0,
                    "max": None,
                },
            },
        }

        summary, fitter = test_helpers.run_bindfit(input_file, hostname, guestname, fitter_name, method, normalise, flavour, dilute, params)

        #K = 334 +/- 2.5
        test_helpers.assertValueInRange(self, summary["fit"]["params"]["k"]["value"], 334, 2.5)

    def test_nmr_1to2(self):
        input_file = "tests/nmr1to2/NMR1to2.csv"
        hostname = "Host"
        guestname = "Guest"
        fitter_name = "nmr1to2"
        method = "Nelder-Mead"
        normalise = True
        dilute = False
        flavour = "none"
        params = {
            "k11": {
                    "init": 100.0,
                    "bounds": {
                        "min": 0.0,
                        "max": None,
                    },
                },
            "k12": {
                "init": 100.0,
                "bounds": {
                    "min": 0.0,
                    "max": None,
                },
            },
        }

        summary, fitter = test_helpers.run_bindfit(input_file, hostname, guestname, fitter_name, method, normalise, flavour, dilute, params)

        #K11 = 13503 +/- 25
        #K12 = 413 +/- 15
        test_helpers.assertValueInRange(self, summary["fit"]["params"]["k11"]["value"], 13503, 25)
        test_helpers.assertValueInRange(self, summary["fit"]["params"]["k12"]["value"], 413, 15)
    
    def test_molefracs(self):
        rng = np.random.default_rng(0)
        for i in range(5):
            h0 = rng.uniform(1e-6, 1e-3, 10)
            g0 = rng.uniform(1e-6, 1e-3, 10)
            k11 = rng.uniform(1e3, 1e6)
            k12 = rng.uniform(1e2, 1e6)
            k13 = rng.uniform(1e2, 1e6)

            with self.subTest(function="uv_1to1", number=i):
                #UV 1:1
                fit, disp = fc.uv_1to1([k11], (h0, g0))
                H, HG = fit
                assert np.allclose(H + HG, h0, rtol=1e-6, atol=1e-12)
                assert np.allclose(disp.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest(function="uv_2to1", number=i):
                #UV 2:1
                fit, disp = fc.uv_2to1([k11, k12], (h0, g0))
                H, HG, H2G = fit
                assert np.allclose(H + HG + H2G, h0, rtol=1e-6, atol=1e-12)
                assert np.allclose(disp.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("uv_1to2", number=i):
                #UV 1:2
                fit, disp = fc.uv_1to2([k11, k12], (h0, g0))
                H, HG, HG2 = fit
                assert np.allclose(H + HG + HG2, h0, rtol=1e-6, atol=1e-12)
                assert np.allclose(disp.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("uv_3to1", number=i):
                #UV 3:1
                fit, disp = fc.uv_3to1([k11, k12, k13], (h0, g0))
                H, HG, H2G, H3G = fit
                assert np.allclose(H + HG + H2G + H3G, h0, rtol=1e-6, atol=1e-12)
                assert np.allclose(disp.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("uv_1to3", number=i):
                #UV 1:3
                fit, disp = fc.uv_1to3([k11, k12, k13], (h0, g0))
                H, HG, HG2, HG3 = fit
                #print(fit)
                assert np.allclose(H + HG + HG2 + HG3, h0, rtol=1e-6, atol=1e-12)
                assert np.allclose(disp.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("nmr_1to1", number=i):
                #NMR 1:1
                fit, disp = fc.nmr_1to1([k11], (h0, g0))
                assert np.allclose(fit.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("nmr_2to1", number=i):
                #NMR 2:1
                fit, disp = fc.nmr_2to1([k11, k12], (h0, g0))
                assert np.allclose(fit.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("nmr_3to1", number=i):
                #NMR 3:1
                fit, disp = fc.nmr_3to1([k11, k12, k13], (h0, g0))
                assert np.allclose(fit.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("nmr_1to2", number=i):
                #NMR 1:2
                fit, disp = fc.nmr_1to2([k11, k12], (h0, g0))
                assert np.allclose(fit.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

            with self.subTest("nmr_1to3", number=i):
                #NMR 1:3
                fit, disp = fc.nmr_1to3([k11, k12, k13], (h0, g0))
                assert np.allclose(fit.sum(axis=0), 1, rtol=1e-9)
                assert np.all(fit >= 0)
                assert np.all(disp >= 0)

if __name__ == '__main__':
    unittest.main()