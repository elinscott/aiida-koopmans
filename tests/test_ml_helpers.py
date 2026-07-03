"""Unit tests for :mod:`aiida_koopmans.ml_helpers`.

Pure-python/numpy tests — no AiiDA profile, no QE runs. Where scikit-learn
is importable, the closed-form estimators are cross-checked against the
exact sklearn stack the legacy code used (``StandardScaler`` +
``Ridge(alpha=1.0)``, ``Ridge(alpha=0.0)``, ``DummyRegressor('mean')``).
"""

# ruff: noqa: E741, N806
# (physics / ML notation: ``l`` angular momentum, ``X`` / ``X_test`` design matrices.)

from __future__ import annotations

import numpy as np
import pytest

from aiida_koopmans import ml_helpers

# ----------------------------------------------------------------------
# Radial basis precomputation
# ----------------------------------------------------------------------


class TestRadialBasis:
    def test_betas_orthonormalize_overlap(self):
        """beta^T S beta = identity — the defining property of the Löwdin betas."""
        n_max, l_max = 3, 2
        alphas, betas = ml_helpers.precompute_parameters_of_radial_basis(n_max, l_max, 0.5, 4.0)
        for l in range(l_max + 1):
            s = ml_helpers.compute_s(n_max, l, alphas)
            beta = betas[:, :, l]
            np.testing.assert_allclose(beta.T @ s @ beta, np.eye(n_max), atol=1e-8)

    def test_alphas_enforce_decay_threshold(self):
        """phi_nl(r_thr) == thr by construction of the decay coefficients."""
        n_max, l_max, thr = 4, 3, 1e-3
        r_thrs = np.linspace(1.0, 4.0, n_max)
        alphas = ml_helpers.compute_alphas(n_max, l_max, r_thrs, thr)
        for n in range(n_max):
            for l in range(l_max + 1):
                phi_at_thr = ml_helpers.phi(np.array(r_thrs[n]), l, alphas[n, l])
                assert phi_at_thr == pytest.approx(thr, rel=1e-10)

    def test_shapes(self):
        alphas, betas = ml_helpers.precompute_parameters_of_radial_basis(4, 4, 0.5, 4.0)
        assert alphas.shape == (4, 5)
        assert betas.shape == (4, 4, 5)


# ----------------------------------------------------------------------
# Spherical harmonics
# ----------------------------------------------------------------------


class TestRealSphericalHarmonics:
    def test_y00_is_constant(self):
        theta = np.array([[[0.3]]])
        phi_angle = np.array([[[1.2]]])
        y = ml_helpers.real_spherical_harmonics(theta, phi_angle, 0, 0)
        assert y[0, 0, 0] == pytest.approx(0.5 / np.sqrt(np.pi))

    def test_y10_is_cos_theta(self):
        theta = np.array([[[0.7]]])
        phi_angle = np.array([[[0.4]]])
        y = ml_helpers.real_spherical_harmonics(theta, phi_angle, 1, 0)
        expected = np.sqrt(3 / (4 * np.pi)) * np.cos(0.7)
        assert y[0, 0, 0] == pytest.approx(expected)


# ----------------------------------------------------------------------
# Power spectrum
# ----------------------------------------------------------------------


class TestPowerSpectrum:
    def test_minimal_case_analytic(self):
        """n_max=1, l_max=0: power = [c_orb^2, c_orb*c_tot, c_tot^2]."""
        power = ml_helpers.compute_power_spectrum(
            np.array([2.0]), np.array([3.0]), n_max=1, l_max=0
        )
        np.testing.assert_allclose(power, [4.0, 6.0, 9.0])

    def test_length_formula(self):
        """len(power) = 3 * n_max(n_max+1)/2 * (l_max+1) (orb-orb, orb-tot, tot-tot blocks)."""
        n_max, l_max = 4, 3
        n_coeffs = n_max * sum(2 * l + 1 for l in range(l_max + 1))
        rng = np.random.default_rng(0)
        power = ml_helpers.compute_power_spectrum(
            rng.normal(size=n_coeffs), rng.normal(size=n_coeffs), n_max=n_max, l_max=l_max
        )
        assert len(power) == 3 * (n_max * (n_max + 1) // 2) * (l_max + 1)

    def test_m_summation(self):
        """l_max=1: the m-components of l=1 are summed within one power entry."""
        # n_max=1, l_max=1 -> 4 coefficients per density: (l=0,m=0), (l=1, m=-1,0,1)
        c_orb = np.array([1.0, 2.0, 3.0, 4.0])
        c_tot = np.zeros(4)
        power = ml_helpers.compute_power_spectrum(c_orb, c_tot, n_max=1, l_max=1)
        # orb-orb block: l=0 entry then l=1 entry; tot blocks are all zero.
        np.testing.assert_allclose(power, [1.0, 4.0 + 9.0 + 16.0, 0.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# XML density parsing
# ----------------------------------------------------------------------


def _make_density_xml(tag: str, array: np.ndarray) -> str:
    """Serialise a (nz, ny, nx) periodic array the way bin2xml lays it out."""
    nz, ny, nx = array.shape
    # bin2xml grids are periodic-endpoint inclusive: nr counts the unique
    # points, the parser adds +1 back on.
    z_blocks = []
    for k in range(nz):
        values = " ".join(str(array[k, j, i]) for j in range(ny) for i in range(nx))
        z_blocks.append(f"<z.{k + 1}>{values}</z.{k + 1}>")
    return (
        f"<ROOT><{tag}><INFO nr1='{nx}' nr2='{ny}' nr3='{nz}'/>"
        + "".join(z_blocks)
        + f"</{tag}></ROOT>"
    )


class TestReadDensityXml:
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        raw = rng.normal(size=(3, 4, 5))
        xml = _make_density_xml("CHARGE-DENSITY", raw)
        parsed, nr = ml_helpers.read_density_xml(xml, "CHARGE-DENSITY", norm_const=1.0)
        assert nr == (6, 5, 4)
        # The parser drops the periodic wrap point in each dimension.
        assert parsed.shape == (3, 4, 5)
        np.testing.assert_allclose(parsed, raw)

    def test_norm_const_applied(self):
        raw = np.ones((2, 2, 2))
        xml = _make_density_xml("EFFECTIVE-POTENTIAL", raw)
        parsed, _ = ml_helpers.read_density_xml(xml, "EFFECTIVE-POTENTIAL", norm_const=2.5)
        np.testing.assert_allclose(parsed, 2.5)

    def test_missing_tag_raises(self):
        xml = _make_density_xml("CHARGE-DENSITY", np.ones((2, 2, 2)))
        with pytest.raises(ValueError, match="EFFECTIVE-POTENTIAL"):
            ml_helpers.read_density_xml(xml, "EFFECTIVE-POTENTIAL", norm_const=1.0)


# ----------------------------------------------------------------------
# Decomposition (integration-level sanity on a tiny analytic density)
# ----------------------------------------------------------------------


class TestComputeDecomposition:
    def test_gaussian_density_dominated_by_s_channel(self):
        """An isotropic Gaussian at the box centre projects mainly onto l=0."""
        n_grid = 12
        length = 8.0
        axis = np.linspace(0.0, length, n_grid, endpoint=False)
        z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
        center = length / 2
        rho = np.exp(-((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2))

        n_max, l_max = 2, 1
        alphas, betas = ml_helpers.precompute_parameters_of_radial_basis(n_max, l_max, 1.0, 4.0)
        orb_coeffs, tot_coeffs = ml_helpers.compute_decomposition(
            n_max=n_max,
            l_max=l_max,
            r_cut=length / 2,
            total_density_xml=_make_density_xml("CHARGE-DENSITY", rho),
            orbital_densities_xml=[_make_density_xml("EFFECTIVE-POTENTIAL", rho)],
            wannier_centers=[[center, center, center]],
            cell_lengths=[length, length, length],
            alphas=alphas,
            betas=betas,
        )
        assert len(orb_coeffs) == len(tot_coeffs) == 1
        coeffs = ml_helpers.read_coeff_matrix(orb_coeffs[0], tot_coeffs[0], n_max, l_max)
        # s-channel weight dominates the p-channel for an isotropic density.
        s_weight = np.abs(coeffs[0, :, 0, :]).sum()
        p_weight = np.abs(coeffs[0, :, 1, :]).sum()
        assert s_weight > 10 * p_weight
        # Same input density twice -> identical orbital and total coefficients.
        np.testing.assert_allclose(orb_coeffs[0], tot_coeffs[0])


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------


def _synthetic_linear_data(n=20, n_features=3, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)) * np.array([1.0, 10.0, 0.1])
    true_coef = np.array([0.5, -0.02, 3.0])
    y = X @ true_coef + 0.6 + noise * rng.normal(size=n)
    return X, y


class TestEstimators:
    def test_linear_regression_recovers_exact_relation(self):
        X, y = _synthetic_linear_data()
        model = ml_helpers.fit_estimator(X, y, "linear_regression")
        np.testing.assert_allclose(ml_helpers.predict_estimator(model, X), y, atol=1e-10)

    def test_ridge_round_trip_close(self):
        X, y = _synthetic_linear_data()
        model = ml_helpers.fit_estimator(X, y, "ridge_regression")
        pred = ml_helpers.predict_estimator(model, X)
        # alpha=1.0 shrinkage keeps it close but not exact.
        assert np.abs(pred - y).mean() < 0.5

    def test_mean_estimator(self):
        X, y = _synthetic_linear_data()
        model = ml_helpers.fit_estimator(X, y, "mean")
        np.testing.assert_allclose(ml_helpers.predict_estimator(model, X), np.mean(y))

    def test_model_dict_is_json_serialisable(self):
        import json

        X, y = _synthetic_linear_data()
        model = ml_helpers.fit_estimator(X, y, "ridge_regression")
        restored = json.loads(json.dumps(model))
        np.testing.assert_allclose(
            ml_helpers.predict_estimator(restored, X), ml_helpers.predict_estimator(model, X)
        )

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="not implemented"):
            ml_helpers.fit_estimator([[1.0]], [1.0], "gaussian_process")

    def test_constant_feature_does_not_blow_up(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        y = np.array([1.0, 2.0, 3.0])
        model = ml_helpers.fit_estimator(X, y, "ridge_regression")
        assert np.isfinite(ml_helpers.predict_estimator(model, X)).all()

    @pytest.mark.parametrize("estimator_type", ["ridge_regression", "linear_regression", "mean"])
    def test_matches_sklearn(self, estimator_type):
        """Pin numerical equivalence with the exact legacy sklearn stack."""
        sklearn = pytest.importorskip("sklearn")  # noqa: F841

        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y = _synthetic_linear_data(noise=0.3)
        model = ml_helpers.fit_estimator(X, y, estimator_type)
        X_test = _synthetic_linear_data(n=7, seed=7)[0]

        if estimator_type == "ridge_regression":
            scaler = StandardScaler().fit(X)
            ref = Ridge(alpha=1.0).fit(scaler.transform(X), y)
            expected = ref.predict(scaler.transform(X_test))
        elif estimator_type == "linear_regression":
            ref = Ridge(alpha=0.0).fit(X, y)
            expected = ref.predict(X_test)
        else:
            ref = DummyRegressor(strategy="mean").fit(X, y)
            expected = ref.predict(X_test)

        np.testing.assert_allclose(ml_helpers.predict_estimator(model, X_test), expected, atol=1e-8)


# ----------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------


class TestBuildSnapshotDataset:
    def test_closed_shell_pairing(self):
        # 3 filled + 2 empty in the single "none" channel; nspin=2 output has
        # two (identical) spin blocks — only block 0 must be consumed.
        sh = [[-1.0, -2.0, -3.0, -4.0, -5.0], [9.9, 9.9, 9.9, 9.9, 9.9]]
        alphas = {"filled": {"none": [0.6, 0.7, 0.8]}, "empty": {"none": [0.5, 0.4]}}
        ds = ml_helpers.build_snapshot_dataset(sh, alphas)
        assert ds["descriptors"] == [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]]
        assert ds["alphas"] == [0.6, 0.7, 0.8, 0.5, 0.4]
        assert ds["filled"] == [True, True, True, False, False]
        assert ds["labels"] == ["orb_1", "orb_2", "orb_3", "orb_4", "orb_5"]

    def test_spin_polarized_pairing(self):
        sh = [[-1.0, -2.0], [-3.0, -4.0]]
        alphas = {
            "filled": {"up": [0.6], "down": [0.61]},
            "empty": {"up": [0.5], "down": [0.51]},
        }
        ds = ml_helpers.build_snapshot_dataset(sh, alphas)
        # up channel first (spin index 0), then down.
        assert ds["labels"] == ["up_orb_1", "up_orb_2", "down_orb_1", "down_orb_2"]
        assert ds["descriptors"] == [[-1.0], [-2.0], [-3.0], [-4.0]]
        assert ds["alphas"] == [0.6, 0.5, 0.61, 0.51]
        assert ds["filled"] == [True, False, True, False]

    def test_count_mismatch_raises(self):
        sh = [[-1.0, -2.0]]
        alphas = {"filled": {"none": [0.6, 0.7]}, "empty": {"none": [0.5]}}
        with pytest.raises(ValueError, match="mismatch"):
            ml_helpers.build_snapshot_dataset(sh, alphas)

    def test_missing_spin_block_raises(self):
        sh = [[-1.0]]
        alphas = {"filled": {"down": [0.6]}, "empty": {}}
        with pytest.raises(ValueError, match="no matching self-Hartree block"):
            ml_helpers.build_snapshot_dataset(sh, alphas)

    def test_no_channels_raises(self):
        with pytest.raises(ValueError, match="no spin channels"):
            ml_helpers.build_snapshot_dataset([[-1.0]], {"filled": {}, "empty": {}})


class TestConcatenateDatasets:
    def test_merge_prefixes_labels_and_sorts_snapshots(self):
        ds1 = {"descriptors": [[1.0]], "alphas": [0.6], "filled": [True], "labels": ["orb_1"]}
        ds2 = {"descriptors": [[2.0]], "alphas": [0.7], "filled": [False], "labels": ["orb_2"]}
        merged = ml_helpers.concatenate_datasets({"snapshot_2": ds2, "snapshot_1": ds1})
        assert merged["labels"] == ["snapshot_1:orb_1", "snapshot_2:orb_2"]
        assert merged["descriptors"] == [[1.0], [2.0]]
        assert merged["alphas"] == [0.6, 0.7]
        assert merged["filled"] == [True, False]


# ----------------------------------------------------------------------
# Screening model fit / predict / evaluate
# ----------------------------------------------------------------------


def _screening_dataset(n=12, seed=3):
    """Synthetic dataset with distinct occ/emp linear laws alpha(self-Hartree)."""
    rng = np.random.default_rng(seed)
    sh = rng.uniform(-5.0, -1.0, size=n)
    filled = [i % 2 == 0 for i in range(n)]
    alphas = [0.1 * s + (0.9 if f else 0.3) for s, f in zip(sh, filled, strict=True)]
    return {
        "descriptors": [[float(s)] for s in sh],
        "alphas": [float(a) for a in alphas],
        "filled": filled,
        "labels": [f"orb_{i + 1}" for i in range(n)],
    }


class TestScreeningModel:
    def test_fit_predict_round_trip_combined(self):
        ds = _screening_dataset()
        model = ml_helpers.fit_screening_model(ds, "linear_regression", occ_and_emp_together=True)
        # A single linear model can't capture the occ/emp offset exactly, but
        # the fit must be finite and unbiased on average.
        pred = ml_helpers.predict_screening(model, ds)
        metrics = ml_helpers.evaluate_predictions(ds["alphas"], pred)
        assert metrics["n_samples"] == 12
        assert np.isfinite(metrics["rmse"])

    def test_fit_predict_split_recovers_both_laws(self):
        ds = _screening_dataset()
        model = ml_helpers.fit_screening_model(ds, "linear_regression", occ_and_emp_together=False)
        pred = ml_helpers.predict_screening(model, ds)
        metrics = ml_helpers.evaluate_predictions(ds["alphas"], pred)
        # Each submodel sees an exactly linear law -> near-perfect recovery.
        assert metrics["max_abs_error"] < 1e-8

    def test_split_without_empty_orbitals_raises(self):
        ds = _screening_dataset()
        ds["filled"] = [True] * len(ds["filled"])
        with pytest.raises(ValueError, match="no empty orbitals"):
            ml_helpers.fit_screening_model(ds, "linear_regression", occ_and_emp_together=False)

    def test_model_metadata(self):
        ds = _screening_dataset()
        model = ml_helpers.fit_screening_model(
            ds, "ridge_regression", occ_and_emp_together=True, descriptor="self_hartree"
        )
        assert model["descriptor"] == "self_hartree"
        assert model["estimator_type"] == "ridge_regression"
        assert set(model["submodels"]) == {"all"}

    def test_evaluate_predictions_metrics(self):
        metrics = ml_helpers.evaluate_predictions([1.0, 2.0, 3.0], [1.0, 2.5, 2.0])
        assert metrics["mae"] == pytest.approx(0.5)
        assert metrics["max_abs_error"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(np.sqrt((0.0 + 0.25 + 1.0) / 3))
