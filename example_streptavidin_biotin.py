"""Example: streptavidin/biotin as patchy particles using PatchyTable.

This sets up a minimal two-type HOOMD simulation (streptavidin type "S",
biotin type "B") with a tabulated, patch-modulated pair potential between
them, and runs a short Langevin trajectory.

The (r, U) table below is a PLACEHOLDER (a WCA repulsive core + a Gaussian
attractive well) standing in for your real tabulated binding-pocket data --
replace `build_placeholder_table()` with e.g.:

    r, U = np.loadtxt("streptavidin_biotin_table.txt", unpack=True)

Like patchy_table.py itself, this script has not been run against a real
HOOMD install in this session (no Python/HOOMD available here) -- read it
as a structurally-checked sketch (API calls cross-referenced against the
HOOMD-blue source) rather than a tested example. Run it yourself and watch
for errors/exceptions before trusting the physics.
"""

import numpy as np
import hoomd

# PatchyTable is the Custom force class defined in patchy_table.py, which
# must be importable from wherever this script runs (same directory works).
from patchy_table import PatchyTable


def build_placeholder_table(r_min=0.5, r_cut=3.0, n_points=200,
                             sigma=1.0, epsilon=1.0, well_center=1.2,
                             well_width=0.3, well_depth=3.0):
    """WCA repulsive core + Gaussian attractive well, purely illustrative.

    Replace with your real tabulated (r, U) data -- e.g. via
    np.loadtxt(...) -- before using this for anything real. The Patchy
    functional form needs a genuinely repulsive core here to keep patch-
    aligned particles from collapsing into each other.
    """
    # Sample the radial potential on a grid from r_min out to the cutoff --
    # this is exactly the array format PatchyTable expects: (r, U) pairs
    # fed into a cubic spline internally.
    r = np.linspace(r_min, r_cut, n_points)

    # WCA = Weeks-Chandler-Andersen: the purely-repulsive half of a
    # Lennard-Jones potential, shifted up by epsilon so it goes to exactly
    # 0 (not -epsilon) at its cutoff, then hard-zeroed beyond that cutoff.
    # This is the "don't let particles overlap" part of the table.
    wca = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6) + epsilon
    wca[r > sigma * 2 ** (1 / 6)] = 0.0

    # A Gaussian dip centered at well_center models the actual "binding"
    # attraction -- deep and narrow here (well_depth, well_width) as a
    # stand-in for a strong, specific interaction like streptavidin-biotin.
    well = -well_depth * np.exp(-0.5 * ((r - well_center) / well_width) ** 2)

    # Sum the two pieces: repulsive at short range, attractive around
    # well_center, ~0 out at r_cut.
    return r, wca + well


def build_snapshot():
    # A Snapshot is HOOMD's plain-data description of one simulation frame
    # (positions, types, box, ...) used to initialize a Simulation's state.
    snapshot = hoomd.Snapshot()

    # In MPI runs, only rank 0 holds/sets the full snapshot data; guarding
    # with this check is required even though we're not using MPI here, so
    # the script also works unmodified under mpirun.
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = 2
        snapshot.particles.types = ["S", "B"]
        # typeid indexes into the types list above: 0 -> "S", 1 -> "B".
        snapshot.particles.typeid[:] = [0, 1]

        # Start them separated along x, facing each other.
        snapshot.particles.position[:] = [[-1.0, 0, 0], [1.0, 0, 0]]
        # (1, 0, 0, 0) is the identity quaternion (w, x, y, z) -- both
        # particles start with their body frame aligned to the lab frame.
        snapshot.particles.orientation[:] = [[1, 0, 0, 0], [1, 0, 0, 0]]

        # Nonzero moment of inertia is required for rotational dof to be
        # integrated at all -- HOOMD defaults this to zero.
        snapshot.particles.moment_inertia[:] = [[1.0, 1.0, 1.0],
                                                  [1.0, 1.0, 1.0]]
        snapshot.particles.mass[:] = [1.0, 1.0]

        # [Lx, Ly, Lz, xy, xz, yz] -- a 10x10x10 orthorhombic (untilted) box.
        snapshot.configuration.box = [10, 10, 10, 0, 0, 0]
    return snapshot


def main():
    # CPU device (no GPU needed for this tiny 2-particle example, and this
    # PatchyTable force is pure-Python/CPU-only anyway).
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(build_snapshot())

    # Cell-list neighbor list with a 0.4 buffer -- standard choice; buffer
    # controls how often the neighbor list needs rebuilding as particles move.
    nl = hoomd.md.nlist.Cell(buffer=0.4)

    r, U = build_placeholder_table()
    force = PatchyTable(
        nlist=nl,
        r_cut={("S", "B"): 3.0},
        tables={("S", "B"): (r, U)},
        directors={
            "S": [(1, 0, 0)],   # streptavidin binding-pocket axis
            "B": [(-1, 0, 0)],  # biotin binding-face axis
        },
        envelope_params={
            "S": (np.pi / 6, 20.0),   # narrow, steep pocket patch
            "B": (np.pi / 4, 20.0),
        },
    )

    # Langevin thermostat/integrator: applies drag + random thermal kicks
    # to both translational and (since moment_inertia != 0) rotational
    # motion, targeting temperature kT=1.0.
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    integrator = hoomd.md.Integrator(
        dt=0.001,
        integrate_rotational_dof=True,  # required: PatchyTable writes torques
        methods=[langevin],
        forces=[force],
    )
    sim.operations.integrator = integrator

    sim.run(0)  # trigger force computation once before reading energy
    print("initial potential energy:", force.energy)

    sim.run(10_000)
    print("final potential energy:", force.energy)
    print("final positions:\n", sim.state.get_snapshot().particles.position)


if __name__ == "__main__":
    main()
