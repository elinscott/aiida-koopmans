"""Unit tests for :mod:`aiida_koopmans.workgraphs.ml.helpers`.

Pure-python/numpy tests — no AiiDA profile, no QE runs. Where scikit-learn
is importable, the closed-form estimators are cross-checked against the
exact sklearn stack they reproduce (``StandardScaler`` +
``Ridge(alpha=1.0)``, ``Ridge(alpha=0.0)``, ``DummyRegressor('mean')``).
"""

# ruff: noqa: E741, N806
# (physics / ML notation: ``l`` angular momentum, ``X`` / ``X_test`` design matrices.)

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

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

    def test_degenerate_decay_radii_raise(self):
        """Coincident r_min/r_max collapse the decay radii, making the overlap singular."""
        with pytest.raises(ValueError, match="Failed to precompute the radial basis"):
            ml_helpers.precompute_parameters_of_radial_basis(3, 1, 1.0, 1.0)


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

    def test_missing_info_element_raises(self):
        xml = "<ROOT><CHARGE-DENSITY><z.1>1.0</z.1></CHARGE-DENSITY></ROOT>"
        with pytest.raises(ValueError, match="no <INFO> element"):
            ml_helpers.read_density_xml(xml, "CHARGE-DENSITY", norm_const=1.0)

    def test_incomplete_info_attributes_raises(self):
        xml = '<ROOT><CHARGE-DENSITY><INFO nr1="2" nr2="2"/><z.1>1.0</z.1></CHARGE-DENSITY></ROOT>'
        with pytest.raises(ValueError, match="lacks nr1/nr2/nr3"):
            ml_helpers.read_density_xml(xml, "CHARGE-DENSITY", norm_const=1.0)


class TestParseXmlArray:
    def test_missing_z_element_raises(self):
        xml = _make_density_xml("CHARGE-DENSITY", np.ones((2, 2, 2)))
        root = ET.fromstring(xml).find("CHARGE-DENSITY")  # noqa: S314
        root.remove(root.find("z.1"))
        with pytest.raises(ValueError, match="Malformed density xml"):
            ml_helpers.parse_xml_array(root, (3, 3, 3), norm_const=1.0)

    def test_retain_final_element_keeps_the_periodic_point(self):
        raw = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
        xml = _make_density_xml("CHARGE-DENSITY", raw)
        root = ET.fromstring(xml).find("CHARGE-DENSITY")  # noqa: S314
        stripped = ml_helpers.parse_xml_array(root, (3, 3, 3), norm_const=1.0)
        retained = ml_helpers.parse_xml_array(
            root, (3, 3, 3), norm_const=1.0, retain_final_element=True
        )
        assert stripped.shape == (2, 2, 2)
        assert retained.shape == (3, 3, 3)
        np.testing.assert_allclose(retained[:-1, :-1, :-1], stripped)


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

    def test_mismatched_orbital_density_and_center_counts_raises(self):
        n_grid = 8
        length = 8.0
        rho = np.ones((n_grid, n_grid, n_grid))
        alphas, betas = ml_helpers.precompute_parameters_of_radial_basis(1, 0, 1.0, 4.0)
        with pytest.raises(ValueError, match="Expected one wannier center per orbital density"):
            ml_helpers.compute_decomposition(
                n_max=1,
                l_max=0,
                r_cut=length / 2,
                total_density_xml=_make_density_xml("CHARGE-DENSITY", rho),
                orbital_densities_xml=[
                    _make_density_xml("EFFECTIVE-POTENTIAL", rho),
                    _make_density_xml("EFFECTIVE-POTENTIAL", rho),
                ],
                wannier_centers=[[4.0, 4.0, 4.0]],
                cell_lengths=[length, length, length],
                alphas=alphas,
                betas=betas,
            )


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

    def test_mismatched_row_counts_raises(self):
        with pytest.raises(ValueError, match="Inconsistent training data"):
            ml_helpers.fit_estimator([[1.0], [2.0], [3.0]], [1.0, 2.0], "ridge_regression")

    def test_constant_feature_does_not_blow_up(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        y = np.array([1.0, 2.0, 3.0])
        model = ml_helpers.fit_estimator(X, y, "ridge_regression")
        assert np.isfinite(ml_helpers.predict_estimator(model, X)).all()

    @pytest.mark.parametrize("estimator_type", ["ridge_regression", "linear_regression", "mean"])
    def test_matches_sklearn(self, estimator_type):
        """Pin numerical equivalence with the exact sklearn stack."""
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
        assert ds["alpha_targets"] == [0.6, 0.7, 0.8, 0.5, 0.4]
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
        assert ds["alpha_targets"] == [0.6, 0.5, 0.61, 0.51]
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

    def test_unrecognised_spin_channel_raises(self):
        sh = [[-1.0]]
        alphas = {"filled": {"spinor": [0.6]}, "empty": {}}
        with pytest.raises(ValueError, match="Unrecognised spin channel"):
            ml_helpers.build_snapshot_dataset(sh, alphas)


class TestConcatenateDatasets:
    def test_merge_prefixes_labels_and_sorts_snapshots(self):
        ds1 = {
            "descriptors": [[1.0]],
            "alpha_targets": [0.6],
            "filled": [True],
            "labels": ["orb_1"],
        }
        ds2 = {
            "descriptors": [[2.0]],
            "alpha_targets": [0.7],
            "filled": [False],
            "labels": ["orb_2"],
        }
        merged = ml_helpers.concatenate_datasets({"snapshot_2": ds2, "snapshot_1": ds1})
        assert merged["labels"] == ["snapshot_1:orb_1", "snapshot_2:orb_2"]
        assert merged["descriptors"] == [[1.0], [2.0]]
        assert merged["alpha_targets"] == [0.6, 0.7]
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
        "alpha_targets": [float(a) for a in alphas],
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
        metrics = ml_helpers.evaluate_predictions(ds["alpha_targets"], pred)
        assert metrics["n_samples"] == 12
        assert np.isfinite(metrics["rmse"])

    def test_fit_predict_split_recovers_both_laws(self):
        ds = _screening_dataset()
        model = ml_helpers.fit_screening_model(ds, "linear_regression", occ_and_emp_together=False)
        pred = ml_helpers.predict_screening(model, ds)
        metrics = ml_helpers.evaluate_predictions(ds["alpha_targets"], pred)
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


class TestDecomposeCrossPower:
    """Cross-power assembly from pw2wannier90 ``wan_mode='decompose'`` output."""

    @staticmethod
    def _qe_orbital_power(coeff, n_max, l_max):
        """Compute the reference orbital-only power, mirroring QE ``wdcp_power_orb``.

        Ordering: outer n1, then n2>=n1, then l; entry = sum_m c(n1,l,m) c(n2,l,m)
        with the flat index ``(n-1)*(l_max+1)^2 + l^2 + m``.
        """
        out = []
        for n1 in range(n_max):
            for n2 in range(n1, n_max):
                for l in range(l_max + 1):
                    ib1 = n1 * (l_max + 1) ** 2 + l**2
                    ib2 = n2 * (l_max + 1) ** 2 + l**2
                    out.append(sum(coeff[ib1 + m] * coeff[ib2 + m] for m in range(2 * l + 1)))
        return np.array(out)

    def test_block_length(self):
        assert ml_helpers.orbital_power_block_length(4, 4) == 5 * 4 * 5 // 2
        assert ml_helpers.orbital_power_block_length(2, 1) == 2 * 2 * 3 // 2

    def test_orbital_power_matches_qe_formula(self):
        """The strongest correctness evidence: our orbital power == QE ``.power``."""
        n_max, l_max = 3, 2
        n_coeff = n_max * (l_max + 1) ** 2
        rng = np.random.default_rng(0)
        coeff = rng.standard_normal(n_coeff)
        got = ml_helpers.orbital_power_from_coefficients(coeff, n_max, l_max)
        ref = self._qe_orbital_power(coeff, n_max, l_max)
        assert got.shape == (ml_helpers.orbital_power_block_length(n_max, l_max),)
        assert np.allclose(got, ref)

    def test_cross_power_orb_block_equals_orbital_power(self):
        n_max, l_max = 2, 2
        n_coeff = n_max * (l_max + 1) ** 2
        rng = np.random.default_rng(2)
        coeff = rng.standard_normal((3, n_coeff))
        group = rng.standard_normal((3, n_coeff))
        power = ml_helpers.cross_power_spectra(coeff, group, n_max, l_max)
        block = ml_helpers.orbital_power_block_length(n_max, l_max)
        # Full descriptor has three blocks (orb-orb, orb-group, group-group).
        assert power.shape == (3, 3 * block)
        for i in range(3):
            assert np.allclose(
                power[i, :block],
                ml_helpers.orbital_power_from_coefficients(coeff[i], n_max, l_max),
            )

    def test_cross_power_group_block_matches_manual(self):
        """The group-group block is the orbital power of the group coefficients."""
        n_max, l_max = 2, 1
        n_coeff = n_max * (l_max + 1) ** 2
        rng = np.random.default_rng(5)
        coeff = rng.standard_normal((1, n_coeff))
        group = rng.standard_normal((1, n_coeff))
        power = ml_helpers.cross_power_spectra(coeff, group, n_max, l_max)
        block = ml_helpers.orbital_power_block_length(n_max, l_max)
        assert np.allclose(
            power[0, 2 * block :],
            self._qe_orbital_power(group[0], n_max, l_max),
        )

    def test_cross_power_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            ml_helpers.cross_power_spectra(np.zeros((2, 8)), np.zeros((2, 9)), 2, 1)

    def test_build_power_spectrum_dataset(self):
        ds = ml_helpers.build_power_spectrum_dataset(
            descriptors=[[1.0, 2.0], [3.0, 4.0]],
            alphas=[0.5, 0.6],
            filled=[True, False],
            labels=["orb_1", "emp_orb_1"],
        )
        assert ds["descriptors"] == [[1.0, 2.0], [3.0, 4.0]]
        assert ds["alpha_targets"] == [0.5, 0.6]
        assert ds["filled"] == [True, False]
        assert ds["labels"] == ["orb_1", "emp_orb_1"]

    def test_build_power_spectrum_dataset_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            ml_helpers.build_power_spectrum_dataset(
                descriptors=[[1.0]], alphas=[0.5, 0.6], filled=[True], labels=["a"]
            )


class TestAssembleOrbitalDensityDataset:
    """Block-to-alpha alignment for the power_spectrum (decompose) route.

    These tests are the discriminators the live-daemon regression will later
    corroborate: blocks carry distinguishable descriptor values so a wrong
    concatenation order (empty-before-filled, down-before-up, or block
    reordering within a slot) changes the asserted row order.
    """

    def test_filled_before_empty_single_channel(self):
        """Discriminator: occ rows must precede emp rows, with matching alphas."""
        block_descriptors = {"occ": [[1.0], [2.0]], "emp": [[10.0], [20.0]]}
        merge_groups = [
            {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
            {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
        ]
        alphas = {"filled": {"none": [0.1, 0.2]}, "empty": {"none": [0.5, 0.6]}}
        ds = ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)
        assert ds["descriptors"] == [[1.0], [2.0], [10.0], [20.0]]
        assert ds["alpha_targets"] == [0.1, 0.2, 0.5, 0.6]
        assert ds["filled"] == [True, True, False, False]
        assert ds["labels"] == ["orb_1", "orb_2", "orb_3", "orb_4"]

    def test_multi_block_within_slot_keeps_group_order(self):
        """Two occupied blocks concatenate in MergeGroup ``blocks`` order."""
        block_descriptors = {"a": [[1.0]], "b": [[2.0], [3.0]]}
        merge_groups = [
            {"filled": True, "spin": "none", "blocks": [{"label": "a"}, {"label": "b"}]},
        ]
        alphas = {"filled": {"none": [0.1, 0.2, 0.3]}, "empty": {}}
        ds = ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)
        assert ds["descriptors"] == [[1.0], [2.0], [3.0]]
        assert ds["filled"] == [True, True, True]

    def test_spin_channels_up_before_down(self):
        """Discriminator: the up channel's orbitals precede the down channel's."""
        block_descriptors = {
            "up_occ": [[1.0]],
            "up_emp": [[2.0]],
            "down_occ": [[11.0]],
            "down_emp": [[12.0]],
        }
        merge_groups = [
            {"filled": True, "spin": "down", "blocks": [{"label": "down_occ"}]},
            {"filled": False, "spin": "up", "blocks": [{"label": "up_emp"}]},
            {"filled": True, "spin": "up", "blocks": [{"label": "up_occ"}]},
            {"filled": False, "spin": "down", "blocks": [{"label": "down_emp"}]},
        ]
        alphas = {
            "filled": {"up": [0.1], "down": [0.3]},
            "empty": {"up": [0.2], "down": [0.4]},
        }
        ds = ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)
        # up (filled, empty) then down (filled, empty), regardless of group list order.
        assert ds["descriptors"] == [[1.0], [2.0], [11.0], [12.0]]
        assert ds["alpha_targets"] == [0.1, 0.2, 0.3, 0.4]
        assert ds["filled"] == [True, False, True, False]
        assert ds["labels"] == ["up_orb_1", "up_orb_2", "down_orb_1", "down_orb_2"]

    def test_length_mismatch_raises(self):
        """A block-WF / alpha count mismatch is a hard error, not silent truncation."""
        block_descriptors = {"occ": [[1.0], [2.0]]}
        merge_groups = [{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}]
        alphas = {"filled": {"none": [0.1]}, "empty": {}}  # 2 WFs vs 1 alpha
        with pytest.raises(ValueError, match="Filled Wannier-function / alpha mismatch"):
            ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)

    def test_missing_block_descriptors_raises(self):
        merge_groups = [{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}]
        alphas = {"filled": {"none": [0.1]}, "empty": {}}
        with pytest.raises(ValueError, match="No descriptors for block `occ`"):
            ml_helpers.assemble_power_spectrum_dataset({}, merge_groups, alphas)

    def test_row_layout_matches_self_hartree_route(self):
        """Parity: labels/filled/alpha order match build_snapshot_dataset exactly.

        For a closed-shell 2-filled/1-empty case the two routes must produce
        identical ``labels`` / ``filled`` / ``alphas`` (only the descriptor
        values differ), so a model trained on either is row-compatible.
        """
        alphas = {"filled": {"none": [0.1, 0.2]}, "empty": {"none": [0.5]}}
        # self_hartree reference (descriptors are the per-orbital SH scalars).
        sh_ref = ml_helpers.build_snapshot_dataset([[7.0, 8.0, 9.0]], alphas)
        # power_spectrum: one occ block (2 WFs) + one emp block (1 WF).
        block_descriptors = {"occ": [[1.0], [2.0]], "emp": [[3.0]]}
        merge_groups = [
            {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
            {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
        ]
        od = ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)
        assert od["labels"] == sh_ref["labels"]
        assert od["filled"] == sh_ref["filled"]
        assert od["alpha_targets"] == sh_ref["alpha_targets"]

    def test_no_channels_raises(self):
        """Empty alphas (no spin channels) is a hard error."""
        with pytest.raises(ValueError, match="no spin channels"):
            ml_helpers.assemble_power_spectrum_dataset({}, [], {"filled": {}, "empty": {}})

    def test_empty_channel_mismatch_raises(self):
        """An empty-orbital / alpha mismatch is caught after the filled check passes."""
        block_descriptors = {"occ": [[1.0]], "emp": [[10.0], [20.0]]}
        merge_groups = [
            {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
            {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
        ]
        # filled matches (1 WF / 1 alpha) so the empty guard (2 WFs / 1 alpha) is what fires.
        alphas = {"filled": {"none": [0.1]}, "empty": {"none": [0.5]}}
        with pytest.raises(ValueError, match="Empty Wannier-function / alpha mismatch"):
            ml_helpers.assemble_power_spectrum_dataset(block_descriptors, merge_groups, alphas)

    def test_non_string_channel_key_raises(self):
        """A non-string spin-channel key is rejected before it corrupts labels."""
        # A stray integer key would otherwise produce a ``"123_orb_1"`` label.
        alphas = {"filled": {123: [0.1]}, "empty": {}}
        with pytest.raises(ValueError, match="Unrecognized spin-channel key"):
            ml_helpers.assemble_power_spectrum_dataset({"occ": [[1.0]]}, [], alphas)


class TestCentresHelpers:
    """``parse_wannier_centres_xyz`` / ``format_group_centres_file`` round-trip."""

    def test_parse_extracts_only_x_pseudospecies_rows(self):
        """Only the ``X`` (Wannier-centre) rows are returned, in file order."""
        xyz = (
            "4\n"
            "comment line\n"
            "X   0.10000000   0.20000000   0.30000000\n"
            "Si  1.00000000   1.00000000   1.00000000\n"
            "X   0.40000000   0.50000000   0.60000000\n"
            "O   2.00000000   2.00000000   2.00000000\n"
        )
        centres = ml_helpers.parse_wannier_centres_xyz(xyz)
        assert centres == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_format_renders_one_triple_per_line_after_comment(self):
        """Each centre becomes one Cartesian triple; the first line is a comment."""
        text = ml_helpers.format_group_centres_file([[0.1, 0.2, 0.3], [4.0, 5.0, 6.0]])
        lines = text.splitlines()
        assert lines[0].startswith("#")
        assert len(lines) == 3
        assert [float(t) for t in lines[1].split()] == [0.1, 0.2, 0.3]
        assert [float(t) for t in lines[2].split()] == [4.0, 5.0, 6.0]

    def test_parse_format_round_trip(self):
        """Formatting then re-parsing (with an ``X`` label) recovers the centres."""
        centres = [[0.123456789, 1.0, -2.5], [3.3, 4.4, 5.5]]
        formatted = ml_helpers.format_group_centres_file(centres)
        xyz = "n\ncomment\n" + "".join(f"X {ln}\n" for ln in formatted.splitlines()[1:])
        recovered = ml_helpers.parse_wannier_centres_xyz(xyz)
        assert np.allclose(recovered, centres)


class TestFinalKiDeltas:
    """Deltas between the computed- and predicted-alpha final KIs."""

    @staticmethod
    def _call(computed_alphas, predicted_alphas, computed_eigs, predicted_eigs):
        from aiida_koopmans.workgraphs.ml.helpers import compute_alpha_and_eigenvalue_deltas

        return compute_alpha_and_eigenvalue_deltas(
            computed_alphas, predicted_alphas, computed_eigs, predicted_eigs
        )

    def test_deltas_are_exact(self):
        """Max and RMS reproduce a hand-computed two-orbital case."""
        computed = {"filled": {"none": [0.60]}, "empty": {"none": [0.70]}}
        predicted = {"filled": {"none": [0.62]}, "empty": {"none": [0.66]}}
        result = self._call(computed, predicted, [[-10.0, -5.0]], [[-10.1, -4.7]])
        assert result["alphas"]["max_abs_delta"] == pytest.approx(0.04)
        assert result["alphas"]["rms_delta"] == pytest.approx((0.5 * (0.02**2 + 0.04**2)) ** 0.5)
        assert result["eigenvalues"]["max_abs_delta"] == pytest.approx(0.3)
        assert result["eigenvalues"]["rms_delta"] == pytest.approx((0.5 * (0.1**2 + 0.3**2)) ** 0.5)
        # Raw rows ride along for reporting.
        assert result["alphas"]["computed"] == [0.60, 0.70]
        assert result["alphas"]["predicted"] == [0.62, 0.66]

    def test_flatten_matches_the_dataset_row_order(self):
        """Flattened rows align with ``build_snapshot_dataset``'s traversal.

        The direct comparison, not flatten-vs-itself: the dataset walks
        channels outer (spin-index order) with filled-then-empty inside,
        and the evaluation's ``predictions`` rows are in that order.
        """
        from aiida_koopmans.workgraphs.ml.helpers import (
            build_snapshot_dataset,
            flatten_alpha_screening,
        )

        alphas = {
            "filled": {"up": [0.6], "down": [0.7]},
            "empty": {"up": [0.5], "down": [0.4]},
        }
        dataset = build_snapshot_dataset([[-1.0, -2.0], [-3.0, -4.0]], alphas)
        assert flatten_alpha_screening(alphas) == dataset["alpha_targets"]
        assert flatten_alpha_screening(alphas) == [0.6, 0.5, 0.7, 0.4]

    def test_empty_payloads_raise_a_domain_error(self):
        """Two empty payloads fail with the domain message, not a numpy error."""
        with pytest.raises(ValueError, match="no orbitals"):
            self._call({"filled": {}, "empty": {}}, {"filled": {}, "empty": {}}, [[]], [[]])

    def test_layout_mismatch_raises(self):
        """Twin arms that cover different orbital counts are rejected."""
        with pytest.raises(ValueError, match="did not share a layout"):
            self._call(
                {"filled": {"none": [0.6, 0.6]}, "empty": {}},
                {"filled": {"none": [0.6]}, "empty": {}},
                [[0.0]],
                [[0.0]],
            )

    def test_eigenvalue_shape_mismatch_raises(self):
        """Twin arms whose eigenvalue arrays disagree in shape are rejected."""
        with pytest.raises(ValueError, match="shapes"):
            self._call(
                {"filled": {"none": [0.6]}, "empty": {}},
                {"filled": {"none": [0.6]}, "empty": {}},
                [[0.0, 1.0]],
                [[0.0]],
            )


# ----------------------------------------------------------------------
# Radial-basis stamp: fit, resolve, and the checks that guard prediction
# ----------------------------------------------------------------------


def _dataset(width=1):
    return {
        "descriptors": [[float(i)] * width for i in (1, 2, 3)],
        "alpha_targets": [0.5, 0.6, 0.7],
        "filled": [True, True, False],
        "labels": ["orb_1", "orb_2", "orb_3"],
    }


def _power_spectrum_model(width=3, **basis):
    from aiida_koopmans.ml import resolve_radial_basis

    radial_basis = resolve_radial_basis({f"decompose_{key}": value for key, value in basis.items()})
    return ml_helpers.fit_screening_model(
        _dataset(width),
        "linear_regression",
        descriptor="power_spectrum",
        correction="ki",
        init_orbitals="mlwfs",
        radial_basis=radial_basis,
    )


class TestRadialBasisResolution:
    def test_unset_keys_fall_back_to_the_calcjob_defaults(self):
        """The stamp records what the pass runs with, not only what was asked."""
        from aiida_koopmans.calculations.pw2wannier_decompose import (
            Pw2wannierDecomposeCalculation,
        )
        from aiida_koopmans.ml import resolve_radial_basis

        resolved = resolve_radial_basis({"decompose_n_max": 6})
        assert resolved["n_max"] == 6
        injected = Pw2wannierDecomposeCalculation._DEFAULTS
        assert resolved["l_max"] == injected["decompose_l_max"]
        assert resolved["r_min"] == injected["decompose_r_min"]
        assert resolved["r_max"] == injected["decompose_r_max"]

    def test_each_setting_resolves_to_its_own_type(self):
        """`RadialBasis` separates the expansion orders from the radii."""
        from aiida_koopmans.ml import resolve_radial_basis

        resolved = resolve_radial_basis(
            {
                "decompose_n_max": "6",
                "decompose_l_max": 6.0,
                "decompose_r_min": 1,
                "decompose_r_max": 4,
            }
        )
        assert resolved == {"n_max": 6, "l_max": 6, "r_min": 1.0, "r_max": 4.0}
        assert isinstance(resolved["n_max"], int)
        assert isinstance(resolved["l_max"], int)
        assert isinstance(resolved["r_min"], float)
        assert isinstance(resolved["r_max"], float)

    def test_mismatch_list_names_only_the_disagreeing_keys(self):
        from aiida_koopmans.ml import radial_basis_mismatches

        left = {"n_max": 6, "l_max": 6, "r_min": 1.0, "r_max": 4.0}
        right = {"n_max": 6, "l_max": 4, "r_min": 1.0, "r_max": 4.0}
        assert radial_basis_mismatches(left, right) == ["l_max"]

    def test_a_missing_stamp_counts_as_a_mismatch(self):
        """A model predating the stamp must not pass by omission."""
        from aiida_koopmans.ml import radial_basis_mismatches

        run_basis = {"n_max": 4, "l_max": 4, "r_min": 0.5, "r_max": 4.0}
        assert radial_basis_mismatches({}, run_basis) == ["n_max", "l_max", "r_min", "r_max"]


class TestModelStamps:
    def test_power_spectrum_fit_stamps_the_basis(self):
        model = _power_spectrum_model(n_max=6, l_max=6, r_min=1.0, r_max=4.0)
        assert (model["n_max"], model["l_max"]) == (6, 6)
        assert (model["r_min"], model["r_max"]) == (1.0, 4.0)

    def test_self_hartree_fit_stamps_no_basis(self):
        """A scalar descriptor has no radial basis; the keys are present but null."""
        model = ml_helpers.fit_screening_model(
            _dataset(), "linear_regression", correction="ki", init_orbitals="kohn-sham"
        )
        assert model["descriptor"] == "self_hartree"
        assert [model[key] for key in ("n_max", "l_max", "r_min", "r_max")] == [None] * 4

    @staticmethod
    def _check(model, **overrides):
        settings = {
            "descriptor": "power_spectrum",
            "correction": "ki",
            "init_orbitals": "mlwfs",
            "radial_basis": {"n_max": 6, "l_max": 6, "r_min": 1.0, "r_max": 4.0},
        }
        settings.update(overrides)
        return ml_helpers.check_model_matches_run(model, **settings)

    def test_a_matching_model_passes(self):
        assert self._check(_power_spectrum_model(n_max=6, l_max=6, r_min=1.0, r_max=4.0)) is None

    def test_a_different_cutoff_radius_raises(self):
        """`r_min` / `r_max` change the basis without changing the row width.

        This is the case that used to return a plausible-looking number
        from a model describing a different quantity.
        """
        from aiida_koopmans.ml import ModelMismatchError

        model = _power_spectrum_model(n_max=6, l_max=6, r_min=0.5, r_max=4.0)
        with pytest.raises(ModelMismatchError, match="r_min") as excinfo:
            self._check(model)
        assert excinfo.value.field == "r_min"

    def test_a_different_expansion_order_raises(self):
        """`n_max` / `l_max` change the row width; the message names the basis."""
        from aiida_koopmans.ml import ModelMismatchError

        model = _power_spectrum_model(n_max=4, l_max=4, r_min=1.0, r_max=4.0)
        with pytest.raises(ModelMismatchError, match="radial basis"):
            self._check(model)

    def test_a_power_spectrum_model_in_a_self_hartree_run_raises(self):
        """The wide-model-fed-a-scalar-row case: caught on the descriptor stamp."""
        from aiida_koopmans.ml import ModelMismatchError

        model = _power_spectrum_model(n_max=6, l_max=6, r_min=1.0, r_max=4.0)
        with pytest.raises(ModelMismatchError, match="trained on `power_spectrum`"):
            self._check(model, descriptor="self_hartree", radial_basis=None)

    def test_an_unstamped_basis_raises(self):
        """A model fitted before the stamp existed cannot be validated, so it is refused."""
        from aiida_koopmans.ml import ModelMismatchError

        model = _power_spectrum_model(n_max=6, l_max=6, r_min=1.0, r_max=4.0)
        for key in ("n_max", "l_max", "r_min", "r_max"):
            del model[key]
        with pytest.raises(ModelMismatchError, match="retrain"):
            self._check(model)


def _orbital(spin, index, filled):
    return {
        "spin": spin,
        "index": index,
        "filled": filled,
        "group_id": index,
        "representative": True,
    }


class TestPowerSpectrumSlots:
    # The down-spin group is listed first and the up-spin empty manifold
    # last, so no consumer can read spin or filling off a slot's position.
    MERGE_GROUPS = [  # noqa: RUF012
        {"filled": True, "spin": "down", "blocks": [{"label": "occ_dn"}]},
        {"filled": True, "spin": "up", "blocks": [{"label": "occ_up"}]},
        {"filled": False, "spin": "up", "blocks": [{"label": "emp_up"}]},
    ]
    BLOCK_DESCRIPTORS = {  # noqa: RUF012
        "occ_dn": [[4.0]],
        "occ_up": [[1.0], [2.0]],
        "emp_up": [[3.0]],
    }
    # Two filled and one empty up-spin orbital, one filled down-spin one.
    ORBITALS = [  # noqa: RUF012
        _orbital("up", 1, True),
        _orbital("up", 2, True),
        _orbital("up", 3, False),
        _orbital("down", 1, True),
    ]

    def test_each_slot_states_its_own_spin_and_filling(self):
        slots = ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS)
        assert slots == [
            {"spin": "down", "filled": True, "rows": [[4.0]]},
            {"spin": "up", "filled": True, "rows": [[1.0], [2.0]]},
            {"spin": "up", "filled": False, "rows": [[3.0]]},
        ]

    def test_one_slot_is_extracted_on_its_own(self):
        assert ml_helpers.power_spectrum_rows_for(
            self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS, spin="up", filled=True
        ) == [[1.0], [2.0]]

    def test_a_block_without_descriptors_raises(self):
        with pytest.raises(ValueError, match="emp_up"):
            ml_helpers.power_spectrum_slots(
                {"occ_up": [[1.0]], "occ_dn": [[4.0]]}, self.MERGE_GROUPS
            )

    def test_rows_follow_the_orbital_s_spin_not_the_slot_s_position(self):
        """The down-spin slot comes first; its row still keys to a down label."""
        slots = ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS)
        assert ml_helpers.power_spectrum_rows_by_orbital(slots, self.ORBITALS) == {
            "up_orb_1": [1.0],
            "up_orb_2": [2.0],
            "up_orb_3": [3.0],
            "down_orb_1": [4.0],
        }

    def test_a_disagreeing_filling_boundary_raises_rather_than_relabelling(self):
        """Two occupied Wannier functions, one occupied orbital: a raise, not a shift.

        Nothing downstream can catch this on counts or labels — the
        channel holds three of each either way — so a row keyed by its
        position in the channel would hand an empty orbital an occupied
        Wannier function's power spectrum, and the occupied estimator
        would score it.
        """
        slots = ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS)
        orbitals = [
            _orbital("up", 1, True),
            _orbital("up", 2, False),
            _orbital("up", 3, False),
            _orbital("down", 1, True),
        ]
        with pytest.raises(ValueError, match="up-spin occupied manifold holds 1"):
            ml_helpers.power_spectrum_rows_by_orbital(slots, orbitals)

    def test_a_manifold_with_the_wrong_count_names_it(self):
        """The supercell case: more orbitals than primitive Wannier functions."""
        slots = ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS)
        orbitals = [*self.ORBITALS, _orbital("up", 4, True)]
        with pytest.raises(ValueError, match="up-spin occupied manifold holds 3"):
            ml_helpers.power_spectrum_rows_by_orbital(slots, orbitals)

    def test_a_manifold_without_descriptors_at_all_raises(self):
        """A screened manifold no block covers must not pass unnoticed."""
        slots = ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS)
        orbitals = [*self.ORBITALS, _orbital("down", 2, False)]
        with pytest.raises(ValueError, match="No descriptors were built for the down-spin empty"):
            ml_helpers.power_spectrum_rows_by_orbital(slots, orbitals)

    def test_row_order_matches_the_dataset_builder(self):
        """The dataset and prediction routes must enumerate orbitals identically.

        A drift between them would train on one ordering and predict on
        another, which no shape check would catch.
        """
        rows = ml_helpers.power_spectrum_rows_by_orbital(
            ml_helpers.power_spectrum_slots(self.BLOCK_DESCRIPTORS, self.MERGE_GROUPS),
            self.ORBITALS,
        )
        dataset = ml_helpers.assemble_power_spectrum_dataset(
            self.BLOCK_DESCRIPTORS,
            self.MERGE_GROUPS,
            {"filled": {"up": [0.1, 0.2], "down": [0.4]}, "empty": {"up": [0.3]}},
        )
        assert [rows[label] for label in dataset["labels"]] == dataset["descriptors"]
