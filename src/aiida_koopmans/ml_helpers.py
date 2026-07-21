"""Pure-Python machine-learning helpers for screening-parameter prediction.

Every function here takes and returns plain Python / numpy data (descriptor
computation and estimator fit/predict), so the ``@task`` wrappers in
:mod:`aiida_koopmans.workgraphs.ml` stay thin wrappers. Sections:

* radial/spherical basis functions
* radial-basis precomputation
* orbital-density decomposition
* power-spectrum construction
* estimators
* dataset assembly / model fit / predict — JSON-serialisable model dicts
  (no sklearn objects), so a trained model can live in an ``orm.Dict``

The estimators reproduce the sklearn stack numerically (``StandardScaler``
+ ``Ridge(alpha=1.0)`` for ridge regression, ``Ridge(alpha=0.0)`` for
linear regression, ``DummyRegressor('mean')`` for the mean estimator) in
closed form with numpy only, so scikit-learn is not a runtime dependency.
scipy (already a transitive dependency) is imported lazily and only by the
orbital-density descriptor path.
"""

# ruff: noqa: E741, N803, N806
# (physics / ML notation: ``l`` is the angular-momentum quantum number,
#  ``X`` / ``Y`` are design-matrix / target arrays.)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict
from xml.etree import ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from aiida_koopmans.types import AlphaScreening

# Bohr radius in Angstrom; the density normalisation (1 / Bohr^3) is
# written against this value.
BOHR_RADIUS_ANG = 0.5291772105638411

ESTIMATOR_TYPES = ("ridge_regression", "linear_regression", "mean")

# Spin-channel key → index into stacked-by-spin arrays (axis 0). Mirrors
# ``aiida_koopmans.types.SpinChannel.axis`` without importing AiiDA here:
# ``none`` (closed shell) and ``up`` share kcp.x's leading spin slot.
_SPIN_KEY_TO_INDEX = {"none": 0, "up": 0, "down": 1}


# ----------------------------------------------------------------------
# Basis functions
# ----------------------------------------------------------------------


def phi(r: np.ndarray | float, l: int, alpha: float) -> Any:
    """Calculate phi_nl from eq. (24) in Himanen et al 2020."""
    return r**l * np.exp(-alpha * r**2)


def g(
    r: np.ndarray, n: int, n_max: int, l: int, betas: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """Calculate g from eq. (23) in Himanen et al 2020."""
    g_vec = np.zeros_like(r)
    g_vec[:, :, :] = sum(
        betas[n_prime, n, l] * phi(r, l, alphas[n_prime, l]) for n_prime in range(n_max)
    )
    return g_vec


def real_spherical_harmonics(
    theta: np.ndarray, phi_angle: np.ndarray, l: int, m: int
) -> np.ndarray:
    """Calculate Y_lm from eq. (20) in Himanen et al 2020.

    ``theta`` is the polar and ``phi_angle`` the azimuthal angle.

    scipy>=1.15 replaced ``sph_harm(m, n, azimuth, polar)`` with
    ``sph_harm_y(n, m, polar, azimuth)`` — the degree/order pair and the
    angle pair both swap places, and calling ``sph_harm_y`` with the old
    argument order silently zeroes every ``l != |m|`` harmonic. The correct
    mapping used here is ``sph_harm(abs(m), l, phi, theta) ==
    sph_harm_y(l, abs(m), theta, phi)``. (The reference ASE-based koopmans
    package currently carries the old-order slip, so descriptor comparisons
    against it will differ until that is fixed upstream.)
    """
    from scipy.special import sph_harm_y

    Y = sph_harm_y(l, abs(m), theta, phi_angle)
    if m < 0:
        Y = np.sqrt(2) * (-1) ** m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1) ** m * Y.real
    return Y.real


# ----------------------------------------------------------------------
# Radial-basis precomputation
# ----------------------------------------------------------------------


def compute_alphas(n_max: int, l_max: int, r_thrs: np.ndarray, thr: float) -> np.ndarray:
    """Compute the decay-coefficients alpha_nl.

    Does this by demanding that for each r_thrs[n], the corresponding phi_nl
    decays to threshold value thr at a cutoff radius of r_thr.
    """
    alphas = np.zeros((n_max, l_max + 1))
    for n in range(n_max):
        for l in range(l_max + 1):
            alphas[n, l] = -1 / r_thrs[n] ** 2 * np.log(thr / (r_thrs[n] ** l))
    return alphas


def compute_overlap(n: int, n_prime: int, l: int, alphas: np.ndarray) -> float:
    """Compute the overlap between two radial basis functions phi_nl, phi_n'l."""
    from scipy.integrate import quad

    def integrand(r: float) -> float:
        return r**2 * phi(r, l, alphas[n, l]) * phi(r, l, alphas[n_prime, l])

    return quad(integrand, 0.0, np.inf)[0]


def compute_s(n_max: int, l: int, alphas: np.ndarray) -> np.ndarray:
    """Compute the overlap matrix S (as in eq. (26) in Himanen et al 2020)."""
    s = np.zeros((n_max, n_max))
    for n in range(n_max):
        for n_prime in range(n_max):
            s[n, n_prime] = compute_overlap(n, n_prime, l, alphas)
    return s


def lowdin(s: np.ndarray) -> np.ndarray:
    """Compute the Löwdin orthogonalization of the matrix s."""
    e, v = np.linalg.eigh(s)
    return np.dot(v / np.sqrt(e), v.T.conj())


def compute_beta(n_max: int, l: int, alphas: np.ndarray) -> np.ndarray:
    """Compute beta making the basis orthogonal (as in eq. (25) Himanen et al 2020)."""
    return lowdin(compute_s(n_max, l, alphas))


def precompute_parameters_of_radial_basis(
    n_max: int, l_max: int, r_min_thr: float, r_max_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the alphas and betas defining the basis functions (as in Himanen et al 2020)."""
    thr = 10 ** (-3)
    r_thrs = np.linspace(r_min_thr, r_max_thr, n_max)
    alphas = compute_alphas(n_max, l_max, r_thrs, thr)
    betas = np.zeros((n_max, n_max, l_max + 1))

    for l in range(l_max + 1):
        betas[:, :, l] = compute_beta(n_max, l, alphas)

    if np.isnan(betas).any() or np.isnan(alphas).any():
        raise ValueError(
            "Failed to precompute the radial basis. "
            "You might want to try a larger `r_min`, e.g. `r_min = 1.0`"
        )

    return alphas, betas


# ----------------------------------------------------------------------
# Orbital-density decomposition
# ----------------------------------------------------------------------


def cart2sph_array(r_cartesian: np.ndarray) -> np.ndarray:
    """Convert an array of cartesian coordinates into the corresponding spherical ones.

    Note that cartesian is z, y, x; spherical is r, theta, phi.
    """
    xy2 = r_cartesian[:, :, :, 2] ** 2 + r_cartesian[:, :, :, 1] ** 2
    r_spherical = np.zeros_like(r_cartesian)
    r_spherical[:, :, :, 0] = np.linalg.norm(r_cartesian, axis=-1)
    r_spherical[:, :, :, 1] = np.arctan2(r_cartesian[:, :, :, 0], np.sqrt(xy2)) + np.pi / 2.0
    r_spherical[:, :, :, 2] = np.arctan2(r_cartesian[:, :, :, 1], r_cartesian[:, :, :, 2]) + np.pi
    return r_spherical


def compute_3d_integral_naive(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute the 3d-integral of an array of functions f on r with a simple trapezoidal rule.

    Note this assumes an orthorhombic cell!
    """
    z = r[:, 0, 0, 0]
    y = r[0, :, 0, 1]
    x = r[0, 0, :, 2]
    return (
        np.sum(f, axis=(0, 1, 2))
        * (x[-1] - x[0])
        * (y[-1] - y[0])
        * (z[-1] - z[0])
        / ((len(x) - 1) * (len(y) - 1) * (len(z) - 1))
    )


def precompute_basis_function(
    r_cartesian: np.ndarray,
    r_spherical: np.ndarray,
    n_max: int,
    l_max: int,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Precompute the total basis (radial * spherical basis function) over the integration domain.

    The basis functions are :func:`g` (parameterised by alphas/betas) and
    :func:`real_spherical_harmonics`.
    """
    # Values of the spherical basis function for each grid point for each (l, m) pair.
    Y_array_all = np.zeros((*np.shape(r_cartesian)[:3], l_max + 1, 2 * l_max + 1))
    for l in range(l_max + 1):
        for i, m in enumerate(range(-l, l + 1)):
            Y_array_all[:, :, :, l, i] = real_spherical_harmonics(
                r_spherical[:, :, :, 1], r_spherical[:, :, :, 2], l, m
            )

    # Values of the radial basis function for each grid point for each (n, l) pair.
    g_array_all = np.zeros((*np.shape(r_cartesian)[:3], n_max, l_max + 1))
    for n in range(n_max):
        for l in range(l_max + 1):
            g_array_all[:, :, :, n, l] = g(r_spherical[:, :, :, 0], n, n_max, l, betas, alphas)

    # Total basis function for each grid point for each (n, l, m); all values
    # corresponding to different m are stored in the last axis.
    number_of_l_elements = sum(2 * l + 1 for l in range(0, l_max + 1))
    total_basis_function_array = np.zeros(
        (*np.shape(r_cartesian)[:3], n_max * number_of_l_elements)
    )
    idx = 0
    for n in range(n_max):
        for l in range(l_max + 1):
            total_basis_function_array[:, :, :, idx : (idx + 2 * l + 1)] = (
                np.expand_dims(g_array_all[:, :, :, n, l], axis=3)
                * Y_array_all[:, :, :, l, 0 : 2 * l + 1]
            )
            idx += 2 * l + 1

    return total_basis_function_array


def get_index(r: np.ndarray, vec: np.ndarray) -> tuple[int, int, int]:
    """Return the index of the array r that is closest to vec."""
    norms = np.linalg.norm(r - vec, axis=3)
    idx_tmp = np.unravel_index(np.argmin(norms), np.shape(r)[:-1])
    return (int(idx_tmp[0]), int(idx_tmp[1]), int(idx_tmp[2]))


def generate_integration_box(
    r: np.ndarray, r_cut: float
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Define the cartesian coordinates of the new integration domain.

    This new integration domain is cubic, has the same grid spacing (dx,dy,dz)
    as the original grid but the mesh-size can be smaller (depending on the
    cutoff value r_cut).
    """
    z = r[:, 0, 0, 0]
    y = r[0, :, 0, 1]
    x = r[0, 0, :, 2]
    dz = z[1] - z[0]
    dy = y[1] - y[0]
    dx = x[1] - x[0]

    nr_new_integration_domain: tuple[int, int, int] = (
        min(int(r_cut / dx), len(x) // 2 - 1),
        min(int(r_cut / dy), len(y) // 2 - 1),
        min(int(r_cut / dz), len(z) // 2 - 1),
    )

    z_ = dz * np.arange(-nr_new_integration_domain[2], nr_new_integration_domain[2] + 1)
    y_ = dy * np.arange(-nr_new_integration_domain[1], nr_new_integration_domain[1] + 1)
    x_ = dx * np.arange(-nr_new_integration_domain[0], nr_new_integration_domain[0] + 1)
    r_new = np.zeros(
        (
            2 * nr_new_integration_domain[2] + 1,
            2 * nr_new_integration_domain[1] + 1,
            2 * nr_new_integration_domain[0] + 1,
            3,
        )
    )
    z_, y_, x_ = np.meshgrid(z_, y_, x_, indexing="ij")
    r_new[:, :, :, 0] = z_
    r_new[:, :, :, 1] = y_
    r_new[:, :, :, 2] = x_

    return nr_new_integration_domain, r_new


def translate_to_new_integration_domain(
    f: np.ndarray,
    wfc_center_index: tuple[int, int, int],
    nr_new_integration_domain: tuple[int, int, int],
) -> np.ndarray:
    """Roll the array f around wfc_center_index, shaped like the integration domain."""
    f_rolled = np.roll(
        f,
        (
            -(wfc_center_index[0] - nr_new_integration_domain[2]),
            -(wfc_center_index[1] - nr_new_integration_domain[1]),
            -(wfc_center_index[2] - nr_new_integration_domain[0]),
        ),
        axis=(0, 1, 2),
    )
    return f_rolled[
        : 2 * nr_new_integration_domain[2] + 1,
        : 2 * nr_new_integration_domain[1] + 1,
        : 2 * nr_new_integration_domain[0] + 1,
    ]


def get_coefficients(
    rho: np.ndarray,
    rho_total: np.ndarray,
    r_cartesian: np.ndarray,
    total_basis_function_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the expansion coefficients of rho / rho_total wrt total_basis_function_array."""
    rho_tmp = np.expand_dims(rho, axis=3)
    rho_total_tmp = np.expand_dims(rho_total, axis=3)

    coefficients = compute_3d_integral_naive(
        rho_tmp * total_basis_function_array, r_cartesian
    ).flatten()
    coefficients_total = compute_3d_integral_naive(
        rho_total_tmp * total_basis_function_array, r_cartesian
    ).flatten()

    return coefficients, coefficients_total


def parse_xml_array(
    xml_root: ET.Element,
    nr: tuple[int, int, int],
    norm_const: float,
    retain_final_element: bool = False,
) -> np.ndarray:
    """Load a charge-density-shaped array from a QE bin2xml Element.

    :param xml_root: the element containing the ``z.<k>`` children (either the
        ``CHARGE-DENSITY`` or the ``EFFECTIVE-POTENTIAL`` element)
    :param nr: the (nr1, nr2, nr3) grid dimensions, periodic-endpoint inclusive
    :param norm_const: normalisation constant (1 / Bohr^3 for densities)
    :param retain_final_element: if True, keep the periodic final element in
        each dimension (xsf format); otherwise strip it
    """
    array_xml = np.zeros((nr[2], nr[1], nr[0]), dtype=float)

    for k in range(nr[2]):
        current_name = "z." + str(k % (nr[2] - 1) + 1)
        entry = xml_root.find(current_name)
        if entry is None or entry.text is None:
            raise ValueError(f"Malformed density xml: missing or empty <{current_name}> element")
        text = entry.text
        rho_tmp = np.array(text.split(), dtype=float)
        for j in range(nr[1]):
            for i in range(nr[0]):
                array_xml[k, j, i] = rho_tmp[(j % (nr[1] - 1)) * (nr[0] - 1) + (i % (nr[0] - 1))]
    array_xml *= norm_const

    if retain_final_element:
        return array_xml
    return array_xml[:-1, :-1, :-1]


def read_density_xml(
    xml_content: str, tag: str, norm_const: float
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Parse one bin2xml density file: return the (stripped) array and the raw grid dims.

    :param xml_content: the full xml file content
    :param tag: ``"CHARGE-DENSITY"`` (total density) or ``"EFFECTIVE-POTENTIAL"``
        (orbital densities, which kcp.x writes under that tag)
    """
    xml_root = ET.fromstring(xml_content)  # noqa: S314
    element = xml_root.find(tag)
    if element is None:
        raise ValueError(f"No <{tag}> element found in xml content")
    info = element.find("INFO")
    if info is None:
        raise ValueError(f"Malformed density xml: <{tag}> has no <INFO> element")
    nr_raw = [info.get(f"nr{i + 1}") for i in range(3)]
    nr = tuple(int(x) + 1 for x in nr_raw if x is not None)
    if len(nr) != 3:
        raise ValueError(f"Malformed density xml: <INFO> lacks nr1/nr2/nr3 attributes ({nr_raw})")
    return parse_xml_array(element, nr, norm_const), nr


def compute_decomposition(
    *,
    n_max: int,
    l_max: int,
    r_cut: float,
    total_density_xml: str,
    orbital_densities_xml: Sequence[str],
    wannier_centers: Sequence[Sequence[float]],
    cell_lengths: Sequence[float],
    alphas: np.ndarray,
    betas: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute the expansion coefficients of the total and orbital densities.

    Inputs are xml strings, per-orbital wannier centers (one per entry of
    ``orbital_densities_xml``, cartesian, same length units as
    ``cell_lengths``) and the orthorhombic cell diagonal ``(a, b, c)``.

    Returns per-orbital ``(orbital_coefficients, total_coefficients)`` lists
    aligned with the input order; feed each pair to
    :func:`compute_power_spectrum`.
    """
    norm_const = 1 / BOHR_RADIUS_ANG**3

    total_density_r, nr_xml = read_density_xml(total_density_xml, "CHARGE-DENSITY", norm_const)

    # Lattice vectors are loaded z-major: lat_vecs = (c, b, a).
    lat_vecs = np.array([cell_lengths[2], cell_lengths[1], cell_lengths[0]])

    # Define the cartesian grid (z, y, x ordering, periodic wrap on the last point).
    r = np.zeros((nr_xml[2] - 1, nr_xml[1] - 1, nr_xml[0] - 1, 3), dtype=float)
    for k in range(nr_xml[2] - 1):
        for j in range(nr_xml[1] - 1):
            for i in range(nr_xml[0] - 1):
                r[k, j, i, :] = np.multiply(
                    np.array(
                        [
                            float(k) / (nr_xml[2] - 1),
                            float(j) / (nr_xml[1] - 1),
                            float(i) / (nr_xml[0] - 1),
                        ]
                    ),
                    lat_vecs,
                )

    # An alternative grid on which the integrations are performed; identical or
    # smaller than the original grid, never larger.
    nr_new_integration_domain, r_cartesian = generate_integration_box(r, r_cut)

    r_spherical = cart2sph_array(r_cartesian)

    # R_nl * Y_lm on every point of the integration domain.
    total_basis_array = precompute_basis_function(
        r_cartesian, r_spherical, n_max, l_max, alphas, betas
    )

    orbital_coefficients: list[np.ndarray] = []
    total_coefficients: list[np.ndarray] = []

    if len(orbital_densities_xml) != len(wannier_centers):
        raise ValueError(
            f"Expected one wannier center per orbital density "
            f"({len(orbital_densities_xml)} densities vs {len(wannier_centers)} centers)"
        )
    for orbital_density_xml, center in zip(orbital_densities_xml, wannier_centers, strict=True):
        rho_r, _ = read_density_xml(orbital_density_xml, "EFFECTIVE-POTENTIAL", norm_const)

        # Bring the density onto the integration domain centred on the
        # orbital's center, folded into the unit cell. The center is
        # (x, y, z) but grids are stored z-major, hence the reversal.
        wfc_center = np.array(
            [center[2] % lat_vecs[0], center[1] % lat_vecs[1], center[0] % lat_vecs[2]]
        )
        center_index = get_index(r, wfc_center)
        rho_r_new = translate_to_new_integration_domain(
            rho_r, center_index, nr_new_integration_domain
        )
        total_density_r_new = translate_to_new_integration_domain(
            total_density_r, center_index, nr_new_integration_domain
        )

        coefficients_orbital, coefficients_total = get_coefficients(
            rho_r_new, total_density_r_new, r_cartesian, total_basis_array
        )
        orbital_coefficients.append(coefficients_orbital)
        total_coefficients.append(coefficients_total)

    return orbital_coefficients, total_coefficients


# ----------------------------------------------------------------------
# Power spectrum
# ----------------------------------------------------------------------


def read_coeff_matrix(
    coff_orb: np.ndarray, coff_tot: np.ndarray, n_max: int, l_max: int
) -> np.ndarray:
    """Read the flat coefficient vectors into a matrix with the correct dimensions."""
    coff_matrix = np.zeros((2, n_max, l_max + 1, 2 * l_max + 1), dtype=float)
    idx = 0
    for n in range(n_max):
        for l in range(l_max + 1):
            for m in range(2 * l + 1):
                coff_matrix[0, n, l, m] = coff_orb[idx]
                coff_matrix[1, n, l, m] = coff_tot[idx]
                idx += 1
    return coff_matrix


def compute_power_mat(coff_matrix: np.ndarray, n_max: int, l_max: int) -> np.ndarray:
    """Compute the power spectrum from the coefficient matrices.

    Note that we only store the non-equivalent entries, hence the second
    for-loop of each pair iterates only over indices >= the first's.
    """
    power = []
    for i1, _ in enumerate(["orb", "tot"]):
        for i2 in range(i1, 2):
            for n1 in range(n_max):
                for n2 in range(n1, n_max):
                    for l in range(l_max + 1):
                        power.append(
                            sum(
                                coff_matrix[i1, n1, l, m] * coff_matrix[i2, n2, l, m]
                                for m in range(2 * l + 1)
                            )
                        )
    return np.array(power)


def compute_power_spectrum(
    orbital_coefficients: np.ndarray, total_coefficients: np.ndarray, n_max: int, l_max: int
) -> np.ndarray:
    """Compute one orbital's power-spectrum descriptor."""
    coff_matrix = read_coeff_matrix(orbital_coefficients, total_coefficients, n_max, l_max)
    return compute_power_mat(coff_matrix, n_max, l_max)


# ----------------------------------------------------------------------
# Power spectra from pw2wannier90 ``wan_mode='decompose'`` coefficients
# ----------------------------------------------------------------------
#
# The reciprocal-space decompose feature of pw2wannier90.x writes, per
# Wannier function, an orbital-density coefficient vector (``<seed>_N.coeff``)
# and, about the same centre, a group-density coefficient vector
# (``<seed>_gc_N.coeff``). These two vectors play exactly the roles of the
# legacy "orbital density" and "total density" channels, so the cross-power
# assembly reuses :func:`compute_power_spectrum` unchanged (orbital = channel
# 0, group = channel 1) — the resulting descriptor has the same orb-orb,
# orb-group, group-group block layout as legacy ``compute_power_mat``.


def orbital_power_block_length(n_max: int, l_max: int) -> int:
    """Return the orbital-only power length ``(l_max+1)*n_max*(n_max+1)/2``.

    Matches the length of the QE ``<seed>_N.power`` file and the first block
    of :func:`compute_power_mat` (the ``orb-orb`` channel).
    """
    return (l_max + 1) * n_max * (n_max + 1) // 2


def orbital_power_from_coefficients(
    orbital_coefficients: np.ndarray, n_max: int, l_max: int
) -> np.ndarray:
    """Orbital-only power spectrum from a single ``.coeff`` vector.

    Equals the QE ``<seed>_N.power`` file (and the leading ``orb-orb`` block
    of :func:`compute_power_spectrum`): ``p_{n1 n2 l} = sum_m c(n1,l,m)
    c(n2,l,m)`` for ``n1 <= n2``. Used as the internal-consistency check
    against the binary's own ``.power`` output.
    """
    full = compute_power_spectrum(orbital_coefficients, orbital_coefficients, n_max, l_max)
    return full[: orbital_power_block_length(n_max, l_max)]


def cross_power_spectra(
    coefficients: np.ndarray,
    group_coefficients: np.ndarray,
    n_max: int,
    l_max: int,
) -> np.ndarray:
    """Assemble the per-WF cross-power descriptor from decompose coefficients.

    :param coefficients: ``(num_wann, n_coeff)`` orbital-density coefficients
        (row ``i`` is Wannier function ``i``), the ``coefficients`` array of
        :class:`Pw2wannierDecomposeParser`.
    :param group_coefficients: ``(num_wann, n_coeff)`` group-density
        coefficients about each Wannier centre (the ``group_coefficients``
        array); row ``i`` must be the group density about WF ``i``'s centre.

    Returns a ``(num_wann, n_power_full)`` array whose row ``i`` is the
    orb-orb / orb-group / group-group power spectrum of WF ``i`` in legacy
    ``compute_power_mat`` order.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    group_coefficients = np.asarray(group_coefficients, dtype=float)
    if coefficients.shape != group_coefficients.shape:
        raise ValueError(
            f"orbital and group coefficient arrays must have the same shape, got "
            f"{coefficients.shape} vs {group_coefficients.shape}"
        )
    return np.array(
        [
            compute_power_spectrum(coefficients[i], group_coefficients[i], n_max, l_max)
            for i in range(coefficients.shape[0])
        ],
        dtype=float,
    )


def parse_wannier_centres_xyz(xyz_content: str) -> list[list[float]]:
    """Extract the Wannier-centre coordinates from a wannier90 ``_centres.xyz`` file.

    wannier90's ``write_xyz`` labels the Wannier-function centres with the
    ``X`` pseudo-species (the real atoms follow with their element symbols).
    Returns the ``X`` rows as ``[x, y, z]`` Cartesian-Angstrom triples, in
    file order (i.e. Wannier-function order).
    """
    centres: list[list[float]] = []
    for line in xyz_content.splitlines():
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] == "X":
            centres.append([float(t) for t in tokens[1:4]])
    return centres


def format_group_centres_file(centres: Sequence[Sequence[float]]) -> str:
    """Render centres for pw2wannier90's ``decompose_centres_file``.

    One Cartesian-Angstrom triple per line; the leading ``#`` line is a
    comment the QE reader skips. Passing every Wannier centre here makes the
    group (total) density be decomposed about each orbital's own centre, so
    the resulting ``_gc_N.coeff`` aligns with the orbital ``_N.coeff`` for the
    legacy-comparable cross-power.
    """
    header = "# Wannier centres for the group-density decomposition (Cartesian, Angstrom)\n"
    return header + "".join(f"{c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n" for c in centres)


def build_orbital_density_dataset(
    descriptors: Sequence[Sequence[float]],
    alphas: Sequence[float],
    filled: Sequence[bool],
    labels: Sequence[str],
) -> SnapshotDataset:
    """Assemble a :class:`SnapshotDataset` from aligned per-orbital rows.

    Route-agnostic: the caller supplies aligned per-orbital ``descriptors``
    (e.g. the rows of :func:`cross_power_spectra`), the screening parameters
    ``alphas`` those orbitals were computed to have (kcp.x for DSCF today,
    kcw.x ``screen_parameters`` for DFPT later), the ``filled`` mask, and the
    ``labels``. This keeps the descriptor source and the alpha source
    decoupled, mirroring :func:`build_snapshot_dataset` for the
    ``self_hartree`` route.
    """
    n = len(descriptors)
    if not (len(alphas) == len(filled) == len(labels) == n):
        raise ValueError(
            "descriptors, alphas, filled and labels must be the same length "
            f"({n}, {len(alphas)}, {len(filled)}, {len(labels)})"
        )
    return {
        "descriptors": [[float(x) for x in row] for row in descriptors],
        "alphas": [float(a) for a in alphas],
        "filled": [bool(f) for f in filled],
        "labels": [str(label) for label in labels],
    }


# ----------------------------------------------------------------------
# Estimators (sklearn replaced with closed forms)
# ----------------------------------------------------------------------


def fit_estimator(X: Any, y: Any, estimator_type: str = "ridge_regression") -> dict[str, Any]:
    """Fit an estimator and return it as a JSON-serialisable dict.

    Reproduces these sklearn estimators exactly:

    * ``ridge_regression`` — ``StandardScaler`` + ``Ridge(alpha=1.0)``
    * ``linear_regression`` — ``Ridge(alpha=0.0)`` (ordinary least squares)
    * ``mean`` — ``DummyRegressor(strategy='mean')``

    The returned dict holds ``x_mean`` / ``x_scale`` (identity when no
    scaling), ``coef`` and ``intercept``; prediction is
    ``((x - x_mean) / x_scale) @ coef + intercept``.
    """
    if estimator_type not in ESTIMATOR_TYPES:
        raise ValueError(f"`{estimator_type}` is not implemented as a valid ML estimator.")

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Inconsistent training data: {X.shape[0]} descriptor rows vs {y.shape[0]} targets"
        )
    n_features = X.shape[1]

    if estimator_type == "mean":
        return {
            "estimator_type": estimator_type,
            "x_mean": [0.0] * n_features,
            "x_scale": [1.0] * n_features,
            "coef": [0.0] * n_features,
            "intercept": float(np.mean(y)),
        }

    if estimator_type == "ridge_regression":
        # StandardScaler: population statistics (ddof=0); constant features
        # are left unscaled, matching sklearn's zero-variance handling.
        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale[x_scale == 0.0] = 1.0
        regularization = 1.0
    else:  # linear_regression
        x_mean = np.zeros(n_features)
        x_scale = np.ones(n_features)
        regularization = 0.0

    Xs = (X - x_mean) / x_scale

    # Ridge with an unpenalised intercept: centre, solve, restore.
    xs_offset = Xs.mean(axis=0)
    y_offset = y.mean()
    Xc = Xs - xs_offset
    yc = y - y_offset
    if regularization > 0.0:
        coef = np.linalg.solve(Xc.T @ Xc + regularization * np.eye(n_features), Xc.T @ yc)
    else:
        coef = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    intercept = y_offset - xs_offset @ coef

    return {
        "estimator_type": estimator_type,
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "coef": coef.tolist(),
        "intercept": float(intercept),
    }


def predict_estimator(model: dict[str, Any], X: Any) -> np.ndarray:
    """Predict targets for descriptor rows ``X`` with a :func:`fit_estimator` model."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Xs = (X - np.asarray(model["x_mean"])) / np.asarray(model["x_scale"])
    return Xs @ np.asarray(model["coef"]) + model["intercept"]


# ----------------------------------------------------------------------
# Screening-parameter datasets and models
# ----------------------------------------------------------------------


class SnapshotDataset(TypedDict):
    """Aligned per-orbital training rows for the screening-parameter model.

    Row ``i`` across every list describes one orbital: its descriptor
    vector, the screening parameter it was computed to have, whether it is
    a filled orbital, and its ``orb_<n>`` / ``up_orb_<n>``-style label
    (``snapshot:label`` after :func:`concatenate_datasets`).
    """

    descriptors: list[list[float]]
    alphas: list[float]
    filled: list[bool]
    labels: list[str]


def build_snapshot_dataset(
    self_hartrees: Sequence[Sequence[float]],
    alphas: AlphaScreening,
) -> SnapshotDataset:
    """Pair one snapshot's per-orbital descriptors with its screening parameters.

    :param self_hartrees: per-spin-block self-Hartree lists as parsed from
        kcp.x stdout (``parameters["orbital_data"]["self-Hartree"]``); within
        each spin block the filled orbitals come first, then the empty ones
    :param alphas: the screening parameters the final KI consumed. The spin
        keys are ``SpinChannel`` members (``NONE`` for closed shell, ``UP``
        / ``DOWN`` for spin-polarised); their plain string values, as an
        AiiDA round-trip delivers them, work identically at runtime

    Returns a flat dataset dict with aligned per-orbital lists:
    ``descriptors`` (one row per orbital), ``alphas``, ``filled`` and
    ``labels`` (``orb_<i>`` / ``up_orb_<i>``-style, matching the kcp
    workgraph's orbital keys).
    """
    filled_alphas = alphas.get("filled", {})
    empty_alphas = alphas.get("empty", {})
    channels = sorted(
        set(filled_alphas) | set(empty_alphas),
        key=lambda ch: (_SPIN_KEY_TO_INDEX.get(ch, 0), ch),
    )
    if not channels:
        raise ValueError("No screening parameters provided: `alphas` has no spin channels")

    dataset: SnapshotDataset = {"descriptors": [], "alphas": [], "filled": [], "labels": []}
    for channel in channels:
        try:
            spin_index = _SPIN_KEY_TO_INDEX[channel]
        except KeyError:
            raise ValueError(
                f"Unrecognised spin channel `{channel}` in screening parameters"
            ) from None
        if spin_index >= len(self_hartrees):
            raise ValueError(
                f"Spin channel `{channel}` has no matching self-Hartree block "
                f"(only {len(self_hartrees)} block(s) parsed from kcp.x output)"
            )
        sh_block = list(self_hartrees[spin_index])
        channel_filled = list(filled_alphas.get(channel, []))
        channel_empty = list(empty_alphas.get(channel, []))
        if len(sh_block) != len(channel_filled) + len(channel_empty):
            raise ValueError(
                f"Self-Hartree / alpha mismatch for spin channel `{channel}`: "
                f"{len(sh_block)} self-Hartree values vs "
                f"{len(channel_filled)} filled + {len(channel_empty)} empty alphas"
            )
        # ``.value`` for genuine SpinChannel keys (whose f-string form is
        # "SpinChannel.UP" on Python 3.12+), identity for plain strings.
        prefix = "" if channel == "none" else f"{getattr(channel, 'value', channel)}_"
        for i, (sh, alpha) in enumerate(
            zip(sh_block, [*channel_filled, *channel_empty], strict=True)
        ):
            dataset["descriptors"].append([float(sh)])
            dataset["alphas"].append(float(alpha))
            dataset["filled"].append(i < len(channel_filled))
            dataset["labels"].append(f"{prefix}orb_{i + 1}")
    return dataset


def concatenate_datasets(datasets: Mapping[str, SnapshotDataset]) -> SnapshotDataset:
    """Merge per-snapshot datasets into one, prefixing labels with the snapshot key."""
    merged: SnapshotDataset = {"descriptors": [], "alphas": [], "filled": [], "labels": []}
    for snapshot_label in sorted(datasets):
        dataset = datasets[snapshot_label]
        merged["descriptors"] += list(dataset["descriptors"])
        merged["alphas"] += list(dataset["alphas"])
        merged["filled"] += list(dataset["filled"])
        merged["labels"] += [f"{snapshot_label}:{label}" for label in dataset["labels"]]
    return merged


def fit_screening_model(
    dataset: SnapshotDataset,
    estimator_type: str = "ridge_regression",
    occ_and_emp_together: bool = True,
    descriptor: str = "self_hartree",
) -> dict[str, Any]:
    """Fit the screening-parameter model on a (merged) dataset.

    With ``occ_and_emp_together`` one estimator covers every orbital,
    otherwise separate ``occ`` / ``emp`` estimators are fitted.
    """
    submodels: dict[str, dict[str, Any]] = {}
    if occ_and_emp_together:
        submodels["all"] = fit_estimator(dataset["descriptors"], dataset["alphas"], estimator_type)
    else:
        for key, want_filled in (("occ", True), ("emp", False)):
            rows = [i for i, filled in enumerate(dataset["filled"]) if filled == want_filled]
            if not rows:
                raise ValueError(
                    f"Cannot fit a separate `{key}` model: the training data contains no "
                    f"{'filled' if want_filled else 'empty'} orbitals"
                )
            submodels[key] = fit_estimator(
                [dataset["descriptors"][i] for i in rows],
                [dataset["alphas"][i] for i in rows],
                estimator_type,
            )
    return {
        "descriptor": descriptor,
        "estimator_type": estimator_type,
        "occ_and_emp_together": occ_and_emp_together,
        "submodels": submodels,
    }


def predict_screening(model: dict[str, Any], dataset: SnapshotDataset) -> list[float]:
    """Predict screening parameters for every orbital row of ``dataset``."""
    predictions = np.empty(len(dataset["alphas"]), dtype=float)
    if model.get("occ_and_emp_together", True):
        predictions[:] = predict_estimator(model["submodels"]["all"], dataset["descriptors"])
    else:
        for key, want_filled in (("occ", True), ("emp", False)):
            rows = [i for i, filled in enumerate(dataset["filled"]) if filled == want_filled]
            if rows:
                predictions[rows] = predict_estimator(
                    model["submodels"][key], [dataset["descriptors"][i] for i in rows]
                )
    return predictions.tolist()


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Compute error metrics between computed and predicted screening parameters."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "n_samples": int(errors.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error": float(np.max(np.abs(errors))),
    }
