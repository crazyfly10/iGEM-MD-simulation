import math
import rendering2
import gsd.hoomd
import hoomd
import numpy

N_biotin = 4
N_strep_cons = 4
N_strep_cent = 1
L = 20
types = ["Strep_cent", "Strep_cons", "Biotin"]
typeid = [0] * N_strep_cent + [2] * N_biotin
position = numpy.random.uniform(-L/2, L/2, size=(N_biotin + N_strep_cent, 3))
orientation = [(1, 0, 0, 0)] * (N_biotin + N_strep_cent)
mass = [1.0] * (N_biotin + N_strep_cent)

#calculating the moment intertia of the central particle
initial_cons_pos = numpy.array(
    [[1,0,2], [-1,0,2], [0,1,-2], [0,-1,-2]]
)
particle_mass = 1
particle_radius = 1
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
patches = hoomd.md.pair.aniso.PatchyLJ(nlist=cell, default_r_cut=1.0)
patches.params.default = dict(pair_params=dict(epsilon=0, sigma=1),
                               envelope_params=dict(alpha=math.pi / 4, omega=30))

# theres nothing on the internet for the half pitch angle because ts has not been done before
envelope_params_cons = {'alpha': math.pi/4, 'omega': 30}
pair_params_cons = {'epsilon': 15, 'sigma': 1}
patches.params[('Strep_cons', 'Biotin')] = dict(pair_params=pair_params_cons,
                                 envelope_params=envelope_params_cons)
patches.r_cut[('Strep_cons','Biotin')] = 4
patches.directors.default = []
patches.directors["Strep_cons"] = [(1, 0, 0)]
patches.directors["Biotin"] = [(1, 0, 0)]

# for all other LJ interactions - LJ
LJ = hoomd.md.pair.LJ(nlist=cell, default_r_cut=1.0)
LJ.params.default = dict(epsilon=0, sigma=1)
# biotin biotin
LJ.params[('Biotin', 'Biotin')] = dict(epsilon=1, sigma=1)
LJ.r_cut[('Biotin', 'Biotin')] = 1.1

integrator.forces.append(patches)
integrator.forces.append(LJ)

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
rendering2.render_movie(traj)
#can ad npt if needed
#npt = hoomd
