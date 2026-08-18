"""Tabulated patchy pair potential for HOOMD-blue (hoomd.md.force.Custom).

    U(r_ij, q_i, q_j) = U_table(r) * sum_{m in patches(i)} sum_{n in patches(j)}
                         f(cos th_mi, alpha_i, omega_i) * f(cos th_nj, alpha_j, omega_j)

This mirrors the functional form of hoomd.md.pair.aniso.Patchy (see
hoomd/md/pair/aniso.py in glotzerlab/hoomd-blue trunk), but U_table(r) comes
from user-supplied (r, U) sample points via cubic-spline interpolation
instead of a fixed analytic form (LJ, Mie, Yukawa, ...). No such tabulated
option exists on trunk today, hence this external Custom force.

th_mi is the angle between patch direction m on particle i (director,
rotated into the lab frame by i's orientation quaternion) and
r_hat = (r_j - r_i) / |r_j - r_i|. th_nj is the angle between patch
direction n on particle j and -r_hat.

    f(theta, alpha, omega) = [sigmoid(omega*(cos(theta)-cos(alpha))) - f_min]
                              / (f_max - f_min)
    f_min = sigmoid(omega*(-1 - cos(alpha)))   # value at theta = pi
    f_max = sigmoid(omega*( 1 - cos(alpha)))   # value at theta = 0

WHY THE FORCE ISN'T JUST RADIAL
--------------------------------
U depends on r_i not only through r = |r_j - r_i| but also through
r_hat = (r_j - r_i)/r, which appears in cos(th_mi) and cos(th_nj). So
-dU/dr_i has a tangential component in addition to the usual radial
U'(r) term -- this is what actually reorients particles to align patches.
pair_force_torque() below implements the full gradient, not just the
radial part.

VALIDATION
----------
No HOOMD (or even Python) was available in the session that wrote this, so
none of this has been run against real HOOMD state. Before wiring
PatchyTable into an actual simulation:

  1. Run this file directly: `python patchy_table.py`. It finite-difference
     checks pair_force_torque()'s force and torque against numerical
     gradients of pair_energy() for a random configuration, using only
     numpy (no hoomd/scipy import needed for this check). It should report
     agreement to ~1e-5 or better. This validates the physics formulas in
     isolation from HOOMD's data plumbing.
  2. Separately, sanity-check the HOOMD-facing plumbing (position_with_ghost,
     nlist arrays, torque frame convention) against a case you can compare
     to a known-good result -- e.g. set alpha = pi for both types (patches
     always "on") and check you recover the results of the corresponding
     hoomd.md.pair.Table run, or fit your table to an LJ shape and compare
     against hoomd.md.pair.aniso.PatchyLJ with matching alpha/omega.
  3. The torque array written to cpu_local_force_arrays.torque is assumed
     to be in the LAB (global) frame here -- this matches how
     hoomd.md.force.Constant.constant_torque is documented ("global
     reference frame"), but the generic Force.torques docstring does not
     state this explicitly, and it was not confirmed by inspecting the C++
     backend. Confirm before trusting rotational dynamics.

LIMITATIONS
-----------
* Pure-Python neighbor loop: O(N * avg_neighbors) in Python per step, not
  compiled C++/GPU. This will be much slower than a native HOOMD pair
  potential; fine for prototyping/smaller systems, likely a bottleneck for
  large production runs.
* Assumes a half neighbor list (each unordered pair visited once). This is
  asserted at runtime.
* No support for r_on / xplor smoothing -- only a hard cutoff per type pair,
  matching AnisotropicPair's own restriction.
"""

import numpy as np


def sigmoid(x):
    # Standard logistic sigmoid, squashes (-inf, inf) -> (0, 1).
    # Used below to build a smooth "is this angle inside the patch cone?"
    # indicator that has a well-defined derivative everywhere (a hard cutoff
    # would give zero/undefined force right at the patch edge).
    return 1.0 / (1.0 + np.exp(-x))


def envelope(cos_theta, alpha, omega):
    """f(theta) and df/d(cos_theta), vectorized over cos_theta."""
    # alpha = patch half-angle (radians): how wide the patch cone is.
    # omega = steepness: how sharply f drops from ~1 (inside the patch) to
    # ~0 (outside) as theta crosses alpha. Larger omega = sharper edge.
    cos_alpha = np.cos(alpha)

    # Raw sigmoid(omega*(cos_theta - cos_alpha)) ranges over some interval
    # strictly inside (0, 1) as theta sweeps 0 -> pi, not exactly [0, 1].
    # f_min/f_max are that raw sigmoid's values at the two extremes
    # (theta = pi, i.e. cos_theta = -1, and theta = 0, i.e. cos_theta = 1).
    f_min = sigmoid(omega * (-1.0 - cos_alpha))
    f_max = sigmoid(omega * (1.0 - cos_alpha))

    # s is the raw (unnormalized) sigmoid at the actual angle we were given.
    s = sigmoid(omega * (cos_theta - cos_alpha))

    # Rescale s so f is exactly 0 at theta=pi and exactly 1 at theta=0 --
    # this normalization is what the Patchy docstring's formula does, and
    # it's needed so patches "fully off" really means zero contribution.
    f = (s - f_min) / (f_max - f_min)

    # Analytic derivative of a sigmoid is s*(1-s) times the derivative of
    # its argument (chain rule); argument's derivative w.r.t. cos_theta is
    # omega, and we divide by the same (f_max - f_min) normalization as f.
    # This df_dcos is what feeds the tangential-force and torque formulas
    # in pair_force_torque() below -- we need it, not just f, because force
    # is a *gradient* of energy, and energy depends on cos_theta.
    df_dcos = omega * s * (1.0 - s) / (f_max - f_min)
    return f, df_dcos


def rotate(q, v):
    """Rotate vector(s) v (shape (3,) or (M,3)) by unit quaternion q=(w,x,y,z)."""
    # Each particle's "directors" (patch directions) are defined once, in
    # the particle's own body frame (e.g. "patch points along local +z").
    # As the particle tumbles during the simulation, its orientation
    # quaternion q changes, and we need the patch direction in the lab
    # frame to compare against r_hat. This is the standard formula for the
    # 3x3 rotation matrix equivalent to a unit quaternion (w, x, y, z).
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    # v @ R.T applies R to every row of v at once (v can be one vector or a
    # stack of several patch-direction vectors for a multi-patch particle).
    return v @ R.T


def pair_energy(rij, qi, qj, dirs_i, dirs_j, alpha_i, omega_i, alpha_j,
                 omega_j, U_r):
    """Total patchy pair energy for one (i, j) pair. rij = r_j - r_i."""
    r = np.linalg.norm(rij)
    rhat = rij / r  # unit vector pointing from particle i towards particle j

    # Rotate each particle's body-frame patch directions into the lab frame
    # using its current orientation.
    di_lab = rotate(qi, dirs_i)
    dj_lab = rotate(qj, dirs_j)

    U = 0.0
    # Sum over every (patch on i, patch on j) pair -- a multi-patch particle
    # (e.g. two binding pockets) contributes once per patch combination, as
    # in the Patchy base class's double sum over m, n.
    for di in np.atleast_2d(di_lab):
        # Angle between patch di and the i->j direction.
        cos_ti = di @ rhat
        fi, _ = envelope(cos_ti, alpha_i, omega_i)
        for dj in np.atleast_2d(dj_lab):
            # Angle between patch dj and the j->i direction (-rhat, since
            # rhat points i->j and we want j's view of "towards i").
            cos_tj = dj @ (-rhat)
            fj, _ = envelope(cos_tj, alpha_j, omega_j)
            # Radial potential gated by how "on-patch" both particles are.
            U += U_r * fi * fj
    return U


def pair_force_torque(rij, qi, qj, dirs_i, dirs_j, alpha_i, omega_i, alpha_j,
                       omega_j, U_r, dU_dr):
    """Analytic force on i (F_j = -F_i by translation invariance) and
    torques on i, j (lab frame) for one pair. rij = r_j - r_i.
    """
    r = np.linalg.norm(rij)
    rhat = rij / r
    di_lab = np.atleast_2d(rotate(qi, dirs_i))
    dj_lab = np.atleast_2d(rotate(qj, dirs_j))

    F_i = np.zeros(3)   # translational force on particle i
    T_i = np.zeros(3)   # torque on particle i (lab frame)
    T_j = np.zeros(3)   # torque on particle j (lab frame)
    U_total = 0.0

    for di in di_lab:
        cos_ti = di @ rhat
        fi, dfi_dcos = envelope(cos_ti, alpha_i, omega_i)
        for dj in dj_lab:
            cos_tj = dj @ (-rhat)
            fj, dfj_dcos = envelope(cos_tj, alpha_j, omega_j)

            U_total += U_r * fi * fj

            # --- Force on i = -d(this patch pair's energy)/d(r_i) ---
            #
            # radial part: the familiar "-dU/dr along r_hat" term, just
            # additionally scaled by how on-patch both particles are (fi*fj).
            F_i += dU_dr * fi * fj * rhat

            # tangential part: moving particle i also rotates r_hat itself
            # (even at fixed r), which changes cos_ti and cos_tj, which
            # changes the envelope factors fi, fj. These two lines are that
            # extra contribution -- this is the piece that actually pulls
            # misaligned patches into alignment; without it, particles
            # would only ever attract/repel straight along r_hat and could
            # never rotate towards each other via translational motion.
            F_i += (U_r * fj * dfi_dcos / r) * (di - cos_ti * rhat)
            F_i -= (U_r * fi * dfj_dcos / r) * (dj + cos_tj * rhat)

            # --- Torques: -d(energy)/d(orientation), converted to the
            # standard torque = -(patch direction) x (d(energy)/d(patch
            # direction)) form for a body-fixed vector rotating in a
            # potential. See the module docstring's derivation notes.
            T_i += -U_r * fj * dfi_dcos * np.cross(di, rhat)
            T_j += U_r * fi * dfj_dcos * np.cross(dj, rhat)

    # Newton's third law for the translational part: U depends on i and j
    # only through rij = r_j - r_i, so d(U)/d(r_j) = -d(U)/d(r_i) exactly.
    return F_i, -F_i, T_i, T_j, U_total


def _numeric_check(seed=0):
    """Finite-difference check of pair_force_torque() against pair_energy().
    numpy only -- no hoomd/scipy dependency. Run this file directly.
    """
    rng = np.random.default_rng(seed)

    def random_unit_quat():
        v = rng.normal(size=4)
        return v / np.linalg.norm(v)

    def random_unit_vec():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    # Build one arbitrary, generic two-particle configuration: random
    # positions, random orientations, two patches on i and one on j, so the
    # double-sum-over-patches code path actually gets exercised.
    ri = rng.normal(size=3) * 0.1
    rj = ri + random_unit_vec() * (1.0 + rng.random())
    qi, qj = random_unit_quat(), random_unit_quat()
    dirs_i = np.array([random_unit_vec(), random_unit_vec()])
    dirs_j = np.array([random_unit_vec()])
    alpha_i, omega_i = np.pi / 3, 15.0
    alpha_j, omega_j = np.pi / 4, 10.0

    # A stand-in radial potential (plain LJ) just for this self-test --
    # any smooth U(r) would do, this isn't related to your actual data.
    def U_of_r(r):
        return 4 * ((1 / r) ** 12 - (1 / r) ** 6)

    def dU_of_r(r):
        return 4 * (-12 / r ** 13 + 6 / r ** 7)

    def energy(ri, rj, qi, qj):
        rij = rj - ri
        r = np.linalg.norm(rij)
        return pair_energy(rij, qi, qj, dirs_i, dirs_j, alpha_i, omega_i,
                            alpha_j, omega_j, U_of_r(r))

    rij = rj - ri
    r = np.linalg.norm(rij)
    F_i, F_j, T_i, T_j, U = pair_force_torque(
        rij, qi, qj, dirs_i, dirs_j, alpha_i, omega_i, alpha_j, omega_j,
        U_of_r(r), dU_of_r(r))

    # --- Force check ---
    # A force is, by definition, F = -dU/dr_i. So nudge particle i by +-h
    # along each axis, measure how the energy changes, and that finite
    # difference should match the analytic F_i we computed above.
    h = 1e-6
    grad_i = np.zeros(3)
    for a in range(3):
        d = np.zeros(3)
        d[a] = h
        grad_i[a] = (energy(ri + d, rj, qi, qj)
                     - energy(ri - d, rj, qi, qj)) / (2 * h)
    print("force check  (should be ~0):",
          np.max(np.abs(F_i - (-grad_i))))

    # --- Torque check ---
    # Analogous idea, but for rotation instead of translation: spin
    # particle i by a tiny angle +-dphi about a random axis (via quaternion
    # multiplication) and see how the energy responds. The component of
    # the analytic torque T_i along that same axis should match
    # -dU/dphi, the same way F_i matched -dU/dr_i above.
    axis = random_unit_vec()

    def quat_from_axis_angle(axis, angle):
        return np.concatenate(([np.cos(angle / 2)], axis * np.sin(angle / 2)))

    def quat_mul(a, b):
        # Hamilton product of two quaternions (w, x, y, z), used to compose
        # a small extra rotation "on top of" the particle's current
        # orientation qi.
        aw, av = a[0], a[1:]
        bw, bv = b[0], b[1:]
        w = aw * bw - av @ bv
        v = aw * bv + bw * av + np.cross(av, bv)
        return np.concatenate(([w], v))

    dphi = h
    qi_plus = quat_mul(quat_from_axis_angle(axis, dphi), qi)
    qi_minus = quat_mul(quat_from_axis_angle(axis, -dphi), qi)
    dU_dphi = (energy(ri, rj, qi_plus, qj)
               - energy(ri, rj, qi_minus, qj)) / (2 * dphi)
    print("torque check (should be ~0):",
          abs(T_i @ axis - (-dU_dphi)))


# hoomd/scipy are only needed to actually run inside a simulation, not to
# run the numpy-only self-check above -- so import them lazily and fall
# back gracefully if they're missing, rather than making this whole file
# unimportable without a HOOMD install.
try:
    import hoomd
    from scipy.interpolate import CubicSpline

    class PatchyTable(hoomd.md.force.Custom):
        """Tabulated-radial x angular-envelope patchy pair force.

        Args:
            nlist (hoomd.md.nlist.NeighborList): attached neighbor list.
            r_cut (dict[tuple[str, str], float]): cutoff radius per type
                pair, e.g. {("S", "B"): 3.0}.
            tables (dict[tuple[str, str], tuple[array, array]]): (r, U)
                sample points per type pair, sorted ascending in r, covering
                at least [r_min, r_cut]. U in energy units.
            directors (dict[str, list[tuple[float, float, float]]]): patch
                direction(s) per particle type, in that particle's BODY
                frame. Normalized automatically; multiple entries mean
                multiple patches on that type.
            envelope_params (dict[str, tuple[float, float]]): (alpha,
                omega) per particle type -- patch half-angle [radians] and
                angular steepness.
        """

        def __init__(self, nlist, r_cut, tables, directors, envelope_params):
            # aniso=True tells HOOMD's integrator this force writes non-zero
            # torques -- see hoomd/md/force.py's Custom docstring. Forgetting
            # this silently drops rotational dynamics rather than erroring.
            super().__init__(aniso=True)
            self.nlist = nlist
            self.nlist.mode = "none"  # no xplor/shift smoothing, hard cutoff

            # frozenset({"S", "B"}) == frozenset({"B", "S"}), so users can
            # write either ("S", "B") or ("B", "S") as a dict key and both
            # this __init__ and set_forces() below will find the same entry.
            self._r_cut = {frozenset(k): v for k, v in r_cut.items()}

            # Precompute one cubic spline (and its derivative, another
            # spline) per type pair, once, instead of re-fitting every
            # simulation step -- set_forces() runs every timestep, so
            # anything reusable belongs here in __init__.
            self._splines = {}
            for k, (r, U) in tables.items():
                spline = CubicSpline(r, U)
                self._splines[frozenset(k)] = (spline, spline.derivative())

            # Normalize every director to a unit vector up front (envelope()
            # assumes cos_theta = dot(unit vector, unit vector)).
            self._directors = {}
            for k, v in directors.items():
                arr = np.asarray(v, dtype=float)
                self._directors[k] = arr / np.linalg.norm(
                    arr, axis=1, keepdims=True)
            self._envelope_params = envelope_params

        def set_forces(self, timestep):
            # These three context managers give zero-copy access to,
            # respectively: particle positions/orientations/types, the
            # neighbor list HOOMD already built this step, and the
            # force/torque/energy buffers we're expected to fill in.
            with self._state.cpu_local_snapshot as snap, \
                 self.cpu_local_nlist_arrays as nl_arrays, \
                 self.cpu_local_force_arrays as force:

                # Our per-pair accumulation below (adding to both i and j,
                # once) is only correct if each unordered pair appears once
                # in the neighbor list -- fail loudly instead of silently
                # double-counting forces if that assumption ever breaks.
                assert nl_arrays.half_nlist, (
                    "PatchyTable assumes a half neighbor list.")

                # "_with_ghost" arrays include HOOMD's periodic/MPI ghost
                # particles, already placed at their correct (unwrapped)
                # image positions -- this is what lets pos[j] - pos[i] give
                # the right minimum-image separation without us having to
                # re-implement periodic wrapping ourselves.
                pos = snap.particles.position_with_ghost
                orient = snap.particles.orientation_with_ghost
                typeid = snap.particles.typeid_with_ghost
                type_names = self._state.particle_types

                # Ghost particles are appended after the real ones, so the
                # (non-"_with_ghost") position array's length is the true
                # local particle count -- what we loop i over below.
                n_local = snap.particles.position.shape[0]

                f_arr = force.force
                t_arr = force.torque
                e_arr = force.potential_energy
                # HOOMD doesn't clear these buffers between steps for us;
                # we own them completely, so zero them before accumulating.
                f_arr[:] = 0
                t_arr[:] = 0
                e_arr[:] = 0

                # CSR-style neighbor list: particle i's neighbors are
                # nlist[head_list[i] : head_list[i] + n_neigh[i]].
                head = nl_arrays.head_list
                n_neigh = nl_arrays.n_neigh
                nlist = nl_arrays.nlist

                for i in range(n_local):
                    ti = type_names[typeid[i]]
                    # Types with no patches defined (e.g. solvent/buffer
                    # particles) just don't participate in this force.
                    if ti not in self._directors:
                        continue
                    alpha_i, omega_i = self._envelope_params[ti]
                    qi = orient[i]

                    start = int(head[i])
                    for k in range(start, start + int(n_neigh[i])):
                        j = int(nlist[k])
                        tj = type_names[typeid[j]]
                        if tj not in self._directors:
                            continue
                        key = frozenset((ti, tj))
                        # No tabulated data for this type combination (e.g.
                        # we only supplied an S-B table, not S-S) -> skip.
                        if key not in self._splines:
                            continue

                        rij = pos[j] - pos[i]
                        r = np.linalg.norm(rij)
                        if r >= self._r_cut[key] or r <= 0:
                            continue

                        spline, dspline = self._splines[key]
                        # Clamp to the table's max r instead of letting the
                        # spline extrapolate past your data -- r < r_cut is
                        # already guaranteed above, but the table itself
                        # might end before r_cut if you built it that way.
                        r_eval = min(r, spline.x[-1])
                        U_r = float(spline(r_eval))
                        dU_dr = float(dspline(r_eval))

                        alpha_j, omega_j = self._envelope_params[tj]
                        qj = orient[j]

                        # All the actual physics lives in the standalone,
                        # finite-difference-tested function above -- this
                        # method is just wiring HOOMD's data into it.
                        F_i, F_j, T_i, T_j, U_pair = pair_force_torque(
                            rij, qi, qj,
                            self._directors[ti], self._directors[tj],
                            alpha_i, omega_i, alpha_j, omega_j,
                            U_r, dU_dr)

                        f_arr[i] += F_i
                        f_arr[j] += F_j
                        t_arr[i] += T_i
                        t_arr[j] += T_j
                        # Split the pair's energy evenly between the two
                        # particles so summing e_arr over all particles
                        # gives the correct total system energy once each
                        # pair (visited once, half-list) is counted.
                        e_arr[i] += 0.5 * U_pair
                        e_arr[j] += 0.5 * U_pair

except ImportError:
    pass  # hoomd/scipy not required to run _numeric_check()


if __name__ == "__main__":
    _numeric_check()


# ---------------------------------------------------------------------------
# Example usage sketch for streptavidin (S) / biotin (B):
#
#   import numpy as np
#   nl = hoomd.md.nlist.Cell(buffer=0.4)
#   r_S_B = np.linspace(0.5, 3.0, 200)          # your tabulated separations
#   U_S_B = ...                                  # your tabulated energies
#   force = PatchyTable(
#       nlist=nl,
#       r_cut={("S", "B"): 3.0},
#       tables={("S", "B"): (r_S_B, U_S_B)},
#       directors={
#           "S": [(0, 0, 1)],   # streptavidin binding-pocket axis
#           "B": [(0, 0, -1)],  # biotin binding-face axis
#       },
#       envelope_params={
#           "S": (np.pi / 6, 20.0),   # narrow, steep pocket patch
#           "B": (np.pi / 4, 20.0),
#       },
#   )
#   sim.operations.integrator.forces.append(force)
# ---------------------------------------------------------------------------
