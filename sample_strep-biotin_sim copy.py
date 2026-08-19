#import itertools
import math
import pandas as pd
import gsd.hoomd
import hoomd
import numpy

def init_biotin_peg_positions(N_PEG, N_biotin, L):
    particle_per_mol = N_PEG+N_biotin
    peg_positions = numpy.zeros((particle_per_mol, 3), dtype=float)

    for i in range(-(particle_per_mol+0.5)//2, (particle_per_mol+0.5)//2):
        x_pos = i * math.cos(math.radians(23.5)) * 0.3321
        z_pos = math.sin(math.radians(23.5)) * (
            math.cos(math.pi * i / 2) ** 2) * 0.3321

        peg_positions[i] = [x_pos, 0.0, z_pos]
    
    peg_positions = numpy.tile(peg_positions, (N_biotin, 1))

    random_loc = numpy.random.uniform(-L/2, L/2, size=(N_biotin, 3))

    counter = 0
    ran_loc_idx = 0
    for idx, current_loc in numpy.ndenumerate(peg_positions):
        if counter < (particle_per_mol/N_biotin):
            counter += 1
        else:
            counter = 0
            ran_loc_idx += 1
        peg_positions[idx][0] += random_loc[ran_loc_idx][0]
        peg_positions[idx][1] += random_loc[ran_loc_idx][1] 
        peg_positions[idx][2] += random_loc[ran_loc_idx][2]

    return peg_positions

N_biotin = 4
N_strep_cons = 4
N_strep_cent = 1
PEG_per_molecule = 10
N_PEG_biotin_mol = 4
N_PEG = N_PEG_biotin_mol * PEG_per_molecule
N_particles = N_biotin + N_strep_cons + N_strep_cent + N_PEG
#spacing = 1.2
#K = math.ceil(N_particles ** (1 / 3))
#L = K * spacing
L = 20
#x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
#position = list(itertools.product(x, repeat=3))
#position = position[0:N_particles]
types = ["Strep_cent", "Strep_cons", "Biotin", "PEG"]
per_molecule_particle_typeid = [0] + [2] + [3] * PEG_per_molecule
typeid = [0] * N_strep_cent + per_molecule_particle_typeid * N_PEG_biotin_mol
position = numpy.random.uniform(-L/2, L/2, size=(N_biotin + N_strep_cent, 3)) + init_biotin_peg_positions(N_PEG_biotin_mol, N_biotin, L)
orientation = [(1, 0, 0, 0)] * (N_biotin + N_strep_cent + N_PEG)
mass = [1.0] * (N_biotin + N_strep_cent + N_PEG)

#calculating the moment intertia of the central particle
initial_cons_pos = numpy.array(
    [[1,0,2], [-1,0,2], [0,1,-2], [0,-1,-2]]
)
particle_mass = 1
particle_radius = 1

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

moment_inertia = [tuple(I_diagonal)] + [(0,0,0)] * N_biotin

# gsd snapshot 
snapshot = gsd.hoomd.Frame()

# particles
snapshot.particles.N = N_biotin + N_strep_cent + N_PEG
snapshot.particles.position = position
snapshot.particles.orientation = orientation
snapshot.particles.typeid = typeid
snapshot.particles.types = types
snapshot.particle.mass = mass
snapshot.particles.moment_inertia = moment_inertia
snapshot.particles.body = [0] * N_strep_cent + [-1] * N_biotin + [-1] * N_PEG
snapshot.configuration.box = [L, L, L, 0, 0, 0]

# bond  
snapshot.bonds.N = PEG_per_molecule * N_PEG_biotin_mol
snapshot.bonds.types = ["Biotin-PEG", "PEG-PEG"]
per_molecule_bond_typeid = [0] + [1] * PEG_per_molecule
snapshot.bonds.typeid = per_molecule_bond_typeid * N_PEG_biotin_mol

molecule_start_index = snapshot.particles.typeid.index(2)
snapshot.bonds.group = numpy.zeros((snapshot.bonds.N, 2), dtype=int)
bond_group_row = 0
for i in range(len(snapshot.particles.typeid[molecule_start_index:])):
    try:
        if snapshot.particles.typeid[i+1] == 2: 
            continue
        else:
            snapshot.bonds.group[bond_group_row] = [molecule_start_index, molecule_start_index + 1]
            molecule_start_index += 1
            bond_group_row += 1
    except IndexError:
        break
del bond_group_row

# angles
snapshot.angles.N = (PEG_per_molecule-1) * N_PEG_biotin_mol
snapshot.angles.types = ["Biotin-PEG-PEG", "PEG-PEG-PEG"]
per_molecule_angle_typeid = [0] + [1] * (PEG_per_molecule-2)
snapshot.angles.typeid = per_molecule_angle_typeid * N_PEG_biotin_mol
snapshot.angles.group = numpy.zeros((snapshot.angles.N, 3), dtype=int)
angle_group_row = 0
for i in range(len(snapshot.particles.typeid[molecule_start_index:])):
    try:
        if snapshot.particles.typeid[i+1] == 2 or snapshot.particles.typeid[i+2] == 2:
            continue
        else:
            snapshot.angles.group[angle_group_row] = [molecule_start_index, molecule_start_index + 1, molecule_start_index + 2]
            molecule_start_index += 1
            angle_group_row += 1
    except IndexError:
        break
del angle_group_row

# dihedrals
snapshot.dihedrals.N = (PEG_per_molecule-2) * N_PEG_biotin_mol
snapshot.dihedrals.types = ["Biotin-PEG-PEG-PEG", "PEG-PEG-PEG-PEG"]
per_molecule_dihedral_typeid = [0] + [1] * (PEG_per_molecule-3)
snapshot.dihedrals.typeid = per_molecule_dihedral_typeid * N_PEG_biotin_mol
snapshot.dihedrals.group = numpy.zeros((snapshot.dihedrals.N, 4), dtype=int)
dihedral_group_row = 0
for i in range(len(snapshot.particles.typeid[molecule_start_index:])):
    try:
        if snapshot.particles.typeid[i+1] == 2 or snapshot.particles.typeid[i+2] == 2 or snapshot.particles.typeid[i+3] == 2:
            continue
        else:
            snapshot.dihedrals.group[dihedral_group_row] = [molecule_start_index, molecule_start_index + 1, molecule_start_index + 2, molecule_start_index + 3]
            molecule_start_index += 1
            dihedral_group_row += 1
    except IndexError:
        break
del dihedral_group_row

with gsd.hoomd.open(name="initial.gsd", mode="x") as f:
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
cell = hoomd.md.nlist.Cell(buffer=0.4, exclusion = ["body","bond","angle","dihedral"])

# for biotin streptavidin constituents - patchyLJ
patches = hoomd.md.pair.ansio.PatchyLJ(nlist=cell)

# theres nothing on the internet for the half pitch angle because ts has not been done before
envelope_params_cons = {'alpha': math.pi/4, 'omega': 30} 
pair_params_cons = {'epsilon': 15, 'sigma': 1}
patches.params[('Strep_cons', 'Biotin')] = dict(pair_params=pair_params_cons,
                                 envelope_params=envelope_params_cons)
patches.r_cut[('Strep_cons','Biotin')] = 4
patches.directors["Strep_cons"] = [(1, 0, 0)]
patches.directors["Biotin"] = [(1, 0, 0)]

# for all other LJ interactions - LJ
LJ = hoomd.md.pair.LJ(nlist=cell)
# biotin biotin
LJ.params[('Biotin', 'Biotin')] = dict(epsilon=1, sigma=1)
LJ.r_cut[('Biotin', 'Biotin')] = 1.1
# strep_centre-strep_centre
LJ.params[('Strep_cent', 'Strep_cent')] = dict(epsilon=0, sigma=1)
# strep_centre-biotin
LJ.params[('Strep_cent', 'Biotin')] = dict(epsilon=0, sigma=1)
# strep_cons-strep_cons
LJ.params[('Strep_cons', 'Strep_cons')] = dict(epsilon=0, sigma=1)
# strep_centre-strep_constituent
LJ.params[('Strep_cent', 'Strep_cons')] = dict(epsilon=0, sigma=1)


df_l = pd.read_csv("peg_bond_length.csv")
df_ba = pd.read_csv("peg_bond_angle.csv")
df_l.loc[df_l.iloc[:, 1].idxmin()]
df_ba.loc[df_ba.iloc[:, 1].idxmin()]

bond_length = df_l.to_numpy().T
bond_angle = pd.read_csv("peg_bond_angle.csv").to_numpy().T
dihedral_angle = pd.read_csv("peg_dihedral_angle.csv").to_numpy().T
dihedral_angle[0] = np.deg2rad(dihedral_angle[0])
bond_angle[0] = np.deg2rad(bond_angle[0])

# Apply the potential on the bonds.
bonds = hoomd.md.bond.Table(bond_length.shape[1])
bonds.params["PEG-PEG"] = dict(r_min = bond_length[0][0], r_max = bond_length[0][-1],
                           U = bond_length[1], F = -numpy.gradient(bond_length[1], bond_length[0]))
# Apply bond angles
angles = hoomd.md.angle.Table(bond_angle.shape[1])
angles.params["PEG-PEG-PEG"] = dict(U = bond_angle[1], tau = -numpy.gradient(bond_angle[1], bond_angle[0]))

# Apply dihedral angles
dihedrals = hoomd.md.dihedral.Table(dihedral_angle.shape[1])
dihedrals.params["PEG-PEG-PEG-PEG"] = dict(U = dihedral_angle[1], tau = -numpy.gradient(dihedral_angle[1], dihedral_angle[0]))

integrator.forces.append(patches)
integrator.forces.append(LJ)
integrator.forces.append(bonds)
integrator.forces.append(angles)
integrator.forces.append(dihedrals)

rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.langevin(
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
del trajectory_writer

traj = gsd.hoomd.open("trajectory.gsd")
#can ad npt if needed
#npt = hoomd 