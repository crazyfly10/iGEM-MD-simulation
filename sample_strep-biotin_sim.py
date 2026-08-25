import math
import rendering2 as rdr
import gsd.hoomd
import hoomd
import numpy

N_biotin = 4
N_strep_cons = 4
N_strep_cent = 1
L = 10
types = ["Strep_cent", "Strep_cons", "Biotin"]
typeid = [0] * N_strep_cent + [2] * N_biotin
position = numpy.vstack([
        numpy.tile(numpy.array([0,0,0]), (N_strep_cent, 1)),
        numpy.random.uniform(-L/2, L/2, size=(N_biotin, 3))]
)
orientation = [(1, 0, 0, 0)] * (N_biotin + N_strep_cent)
mass = [1.0] * N_strep_cent + [244.31] * N_biotin

#calculating the moment intertia of the central particle
initial_cons_pos = numpy.array(
    [[-1.202,-1.202,-1.202], 
     [1.202,1.202,-1.202 ], 
     [-1.202,1.202,1.202], 
     [1.202,-1.202,1.202]]
)
particle_mass = 14500
particle_radius = 1.7
# calcaulating orientation of constituents so their orientation matches the positional offset

I_ref = numpy.array(
    [
        [2 / 5 * particle_mass * particle_radius**2, 0, 0],
        [0, 2 / 5 * particle_mass * particle_radius**2, 0],
        [0, 0, 2 / 5 * particle_mass * particle_radius**2],
    ]
)

# now witht the parallel axis theorem
I_general = numpy.zeros(shape=(3, 3))
for r in initial_cons_pos:
    I_general += I_ref + particle_mass * (
        numpy.dot(r, r) * numpy.identity(3) - numpy.outer(r, r)
    )

# diagonalisation:
I_diagonal, E_vec = numpy.linalg.eig(I_general)
I_diagonal = numpy.real(I_diagonal)
E_vec = numpy.real(E_vec)
R = E_vec.T
new_cons_pos = numpy.dot(R, initial_cons_pos.T).T

# free biotin particles need nonzero rotational inertia so their orientation
# can respond to torques from the patchy potential
I_sphere = 2 / 5 * particle_mass * particle_radius**2
moment_inertia = [tuple(I_diagonal)] * N_strep_cent + [(I_sphere,) * 3] * N_biotin

# gsd snapshot
snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_biotin + N_strep_cent
snapshot.particles.position = position
snapshot.particles.orientation = orientation
snapshot.particles.typeid = typeid
snapshot.particles.types = types
snapshot.particles.mass = mass
snapshot.particles.moment_inertia = moment_inertia
snapshot.configuration.box = [L, L, L, 0, 0, 0]
with gsd.hoomd.open(name="initial.gsd", mode="w") as f:
    f.append(snapshot)

# calcaulating orientation of constituents so their orientation matches the positional offset
def quat_from_a_to_b(a, b):
    a = numpy.array(a, dtype=float)
    b = numpy.array(b, dtype=float)
    a = a / numpy.linalg.norm(a)
    b = b / numpy.linalg.norm(b)

    dot = numpy.dot(a, b)
    if dot < -1 + 1e-8:
        # a and b are anti-parallel - no unique axis, so pick any perpendicular one
        axis = numpy.cross(a, [1, 0, 0])
        if numpy.linalg.norm(axis) < 1e-8:
            axis = numpy.cross(a, [0, 1, 0])
        axis = axis / numpy.linalg.norm(axis)
        return (0.0, *axis)

    w, xyz = 1 + dot, numpy.cross(a, b)
    q = numpy.array([w, *xyz])
    return tuple(q / numpy.linalg.norm(q))


director_rest_direction = (1, 0, 0)
cons_orientations = [quat_from_a_to_b(director_rest_direction, pos) for pos in new_cons_pos]

# defining rigid geometries
rigid = hoomd.md.constrain.Rigid()
rigid.body["Strep_cent"] = {
    "constituent_types": ["Strep_cons", "Strep_cons", "Strep_cons", "Strep_cons"],
    "positions": list(map(tuple, new_cons_pos)),
    "orientations": cons_orientations,
}

# build simulation
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=0)
simulation.create_state_from_gsd(filename="initial.gsd")

rigid.create_bodies(simulation.state) # create the rigid body

# initialise integrator
integrator = hoomd.md.Integrator(dt = 0.002, integrate_rotational_dof=True)
integrator.rigid = rigid




# writing the forces
cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions = ["body"])

# for biotin streptavidin constituents - patchyLJ
# default_r_cut/params.default cover the type pairs that don't interact -
# hoomd requires every type pair to have params/r_cut defined before run()
# mode="shift" shifts U(r_cut) to 0 for the isotropic LJ part (energy
# bookkeeping only - it adds a constant, so it does not change the force
# or the dynamics). Kept on for clean thermodynamics/logging.
patches = hoomd.md.pair.aniso.PatchyLJ(nlist=cell, default_r_cut=1.0, mode="shift")
patches.params.default = dict(pair_params=dict(epsilon=0, sigma=1),
                               envelope_params=dict(alpha=math.pi / 4, omega=30))

# theres nothing on the internet for the half pitch angle because ts has not been done before
envelope_params_cons = {'alpha': math.pi/4, 'omega': 30}

# From a COM-COM PMF calc (biotin bound inside the streptavidin pocket):
# well depth -80 kJ/mol at r = 0.2 nm. epsilon is the well depth directly,
# but sigma is NOT the well location - for 12-6 LJ the minimum sits at
# r_min = 2**(1/6) * sigma, so sigma must be back-solved from the target
# r_min instead of set equal to it.
pmf_r_min = 0.2  # nm, location of the PMF minimum
pmf_depth = 80  # kJ/mol, magnitude of the PMF well depth
sigma_patch = pmf_r_min / 2 ** (1 / 6)
pair_params_cons = {'epsilon': pmf_depth, 'sigma': sigma_patch}
patches.params[('Strep_cons', 'Biotin')] = dict(pair_params=pair_params_cons,
                                 envelope_params=envelope_params_cons)
# r_cut should be set by where the LJ tail has decayed to ~0, not by where
# the PMF calculation happened to stop - the sharp feature near 3.4 nm in
# the PMF is an artifact of the finite sampling range, not a real feature
# of the interaction, so it shouldn't drive this choice.
patches.r_cut[('Strep_cons','Biotin')] = 3 * sigma_patch
patches.directors.default = []
patches.directors["Strep_cons"] = [(1, 0, 0)]
patches.directors["Biotin"] = [(1, 0, 0)]

# for all other LJ interactions - LJ
# mode="shift" only affects energy bookkeeping (adds a constant so
# U(r_cut)=0) - it does not change forces/dynamics, harmless for the
# epsilon=0 default pairs.
LJ = hoomd.md.pair.LJ(nlist=cell, default_r_cut=1.0, mode="shift")
LJ.params.default = dict(epsilon=0, sigma=1)

# biotin-biotin: pure excluded volume (WCA), not a real attractive well -
# free biotin molecules don't chemically stick to each other. The previous
# r_cut=1.1 was smaller than r_min=2**(1/6)*sigma=1.122 (for sigma=1), so
# it was truncating before the potential ever turned attractive anyway -
# made that explicit here instead of leaving it as an accidental near-miss.
# sigma_biotin ~ 0.5 nm approximates real biotin's molecular diameter -
# calibrate if a more precise value is available.
sigma_biotin = 0.5
epsilon_biotin = 1  # kJ/mol scale, soft steric placeholder
LJ.params[('Biotin', 'Biotin')] = dict(epsilon=epsilon_biotin, sigma=sigma_biotin)
LJ.r_cut[('Biotin', 'Biotin')] = sigma_biotin * 2 ** (1 / 6)

# Baseline excluded volume, independent of orientation.
# PatchyLJ's angular envelope multiplies the WHOLE potential - repulsive
# core included - so outside the patch cone (envelope ~ 0) Strep_cons and
# Biotin currently have NO interaction at all and can pass through each
# other. This adds a small always-on repulsive wall to prevent that.
# HOOMD 7.1.2 has no standalone WCA class - WCA is just LJ truncated and
# shifted at its own minimum (r_cut = 2**(1/6) * sigma, mode="shift"),
# which keeps the repulsive branch and drops the attractive well.
# sigma_excl must stay smaller than pmf_r_min (0.2 nm) or this wall would
# sterically block the real bound state - this value is a placeholder and
# should be calibrated against real steric/vdW radii, not derived from data.
wca = hoomd.md.pair.LJ(nlist=cell, default_r_cut=0.0, mode="shift")
wca.params.default = dict(epsilon=0, sigma=1)
sigma_excl = 0.1  # nm, placeholder - must stay < pmf_r_min
epsilon_excl = 1  # kJ/mol scale, matches the PatchyLJ energy convention
wca.params[('Strep_cons', 'Biotin')] = dict(epsilon=epsilon_excl, sigma=sigma_excl)
wca.r_cut[('Strep_cons', 'Biotin')] = sigma_excl * 2 ** (1 / 6)

# Strep_cons-Strep_cons: sized to the actual subunit radius (particle_radius,
# defined above for the rigid-body inertia calc) instead of reusing the tiny
# Strep_cons-Biotin placeholder - two streptavidin subunits should not be
# able to interpenetrate the way the shared placeholder used to allow.
# r_min is set to the sum of the two bead radii (2 * particle_radius), i.e.
# where the spheres just touch.
sigma_strep = 2 * particle_radius / 2 ** (1 / 6)
wca.params[('Strep_cons', 'Strep_cons')] = dict(epsilon=epsilon_excl, sigma=sigma_strep)
wca.r_cut[('Strep_cons', 'Strep_cons')] = sigma_strep * 2 ** (1 / 6)

integrator.forces.append(patches)
integrator.forces.append(LJ)
integrator.forces.append(wca)





rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.Langevin(
    filter=rigid_centers_and_free, kT=1.5
)
integrator.methods.append(langevin)

simulation.operations.integrator = integrator

# periodically write frames to a trajectory over the whole run, instead of
# only writing the final state once the run finishes
trajectory_writer = hoomd.write.GSD(
    filename="trajectory.gsd",
    trigger=hoomd.trigger.Periodic(100),
    mode="wb",
)
simulation.operations.writers.append(trajectory_writer)

simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

simulation.run(10000)
simulation.operations.writers.remove(trajectory_writer)
del trajectory_writer

traj = gsd.hoomd.open("trajectory.gsd")
rdr.render_movie(traj)
#can ad npt if needed
#npt = hoomd
