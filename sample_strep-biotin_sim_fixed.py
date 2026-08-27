import math
#import rendering2 as rdr
import gsd.hoomd
import hoomd
import numpy

N_biotin = 24
N_strep_cons = 4
N_strep_cent = 1
L = 12.5
types = ["Strep_cent", "Strep_cons", "Biotin"]
typeid = [0] * N_strep_cent + [2] * N_biotin

# Eppendorf-tube wall geometry (used both for the wall potential below and
# for sampling biotin's starting positions inside it). Radius/half-height
# kept smaller than L/2=6.25 on purpose - 1.25nm clearance to the box edge
# so the tube stays off the periodic boundary.
TUBE_RADIUS = 5  # nm
TUBE_HALF_HEIGHT = 5  # nm

# Unit convention used throughout this script: length = nm, mass = g/mol
# (amu), energy = kJ/mol. With these three picked, time comes out in ps for
# free - same convention GROMACS uses, no separate time unit to choose.
KB = 0.0083144621  # kJ/(mol*K)
TEMPERATURE_K = 298.0  # room temperature; also used for the solvent viscosity below
kT = KB * TEMPERATURE_K

# Real molar mass/radius of a streptavidin monomer (~13-16 kDa tetramer
# subunit) and of biotin (244.31 g/mol is biotin's actual molar mass).
STREP_SUBUNIT_MASS = 14500  # g/mol, per subunit
STREP_SUBUNIT_RADIUS = 1.7  # nm, per subunit
BIOTIN_MASS = 244.31  # g/mol
BIOTIN_RADIUS = 0.25  # nm

# Sample free biotin starting positions uniformly inside the cylindrical
# well (matching the wall geometry above), not uniformly inside the full
# cubic box - the box corners lie outside the cylinder (a cube of side L
# reaches sqrt(2)*L/2 from the axis, the cylinder only reaches
# TUBE_RADIUS), so cubic sampling could start a biotin already deep inside
# the wall's repulsive core, launching it on step 0.
# Radius uses sqrt(uniform) rather than uniform(0, TUBE_RADIUS) directly -
# needed because points uniform per unit AREA of a disk are NOT uniform in
# r (the area element grows as r dr, so the correct radial CDF is r**2,
# inverted here). BIOTIN_RADIUS is subtracted as a margin so the bead's
# own volume starts clear of the wall surface instead of overlapping it.
_biotin_r = numpy.sqrt(numpy.random.uniform(0, 1, size=N_biotin)) * (TUBE_RADIUS - BIOTIN_RADIUS)
_biotin_theta = numpy.random.uniform(0, 2 * numpy.pi, size=N_biotin)
_biotin_xy = numpy.column_stack([_biotin_r * numpy.cos(_biotin_theta), _biotin_r * numpy.sin(_biotin_theta)])
_biotin_z = numpy.random.uniform(
    -(TUBE_HALF_HEIGHT - BIOTIN_RADIUS), TUBE_HALF_HEIGHT - BIOTIN_RADIUS, size=N_biotin
)
biotin_position = numpy.column_stack([_biotin_xy, _biotin_z])

position = numpy.vstack([
        numpy.tile(numpy.array([0,0,0]), (N_strep_cent, 1)),
        biotin_position]
)
orientation = [(1, 0, 0, 0)] * (N_biotin + N_strep_cent)
# The rigid body's central particle carries the mass of the WHOLE body in
# HOOMD, not a per-particle placeholder - it was previously 1.0 (244x
# lighter than free biotin) instead of the tetramer's real total mass,
# which is why it diffused faster than biotin instead of slower.
mass = [N_strep_cons * STREP_SUBUNIT_MASS] * N_strep_cent + [BIOTIN_MASS] * N_biotin

#calculating the moment intertia of the central particle
initial_cons_pos = numpy.array(
    [[-1.202,-1.202,-1.202],
     [1.202,1.202,-1.202 ],
     [-1.202,1.202,1.202],
     [1.202,-1.202,1.202]]
)
# calcaulating orientation of constituents so their orientation matches the positional offset

I_ref = numpy.array(
    [
        [2 / 5 * STREP_SUBUNIT_MASS * STREP_SUBUNIT_RADIUS**2, 0, 0],
        [0, 2 / 5 * STREP_SUBUNIT_MASS * STREP_SUBUNIT_RADIUS**2, 0],
        [0, 0, 2 / 5 * STREP_SUBUNIT_MASS * STREP_SUBUNIT_RADIUS**2],
    ]
)

# now witht the parallel axis theorem
I_general = numpy.zeros(shape=(3, 3))
for r in initial_cons_pos:
    I_general += I_ref + STREP_SUBUNIT_MASS * (
        numpy.dot(r, r) * numpy.identity(3) - numpy.outer(r, r)
    )

# diagonalisation:
I_diagonal, E_vec = numpy.linalg.eig(I_general)
I_diagonal = numpy.real(I_diagonal)
E_vec = numpy.real(E_vec)
R = E_vec.T
new_cons_pos = numpy.dot(R, initial_cons_pos.T).T

# free biotin particles need nonzero rotational inertia so their orientation
# can respond to torques from the patchy potential - use biotin's own mass
# and radius (this previously reused streptavidin's subunit mass/radius,
# giving biotin ~2700x too much rotational inertia to visibly rotate over a
# short run).
I_biotin_sphere = 2 / 5 * BIOTIN_MASS * BIOTIN_RADIUS**2
moment_inertia = [tuple(I_diagonal)] * N_strep_cent + [(I_biotin_sphere,) * 3] * N_biotin

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
envelope_params_cons = {'alpha': math.pi/12, 'omega': 30}

# Real biotin binds streptavidin with a specific orientation - its ureido
# ring buries into the pocket (H-bonds to Asn23/Ser27/Tyr43) while its
# valeryl tail points back out toward the pocket entrance/loop L3-4 - so
# biotin is NOT free to bind at any orientation, and BOTH beads keep a
# director (mutual alignment required), not just Strep_cons.
#
# The bound-state distance below, however, cannot reuse the atomistic PMF
# value: that -80 kJ/mol well at r=0.2 nm was measured COM-COM between
# biotin and specific pocket residues, not between whole-subunit bead
# centers. With bead radii of 1.7 nm (Strep_cons) and 0.25 nm (Biotin),
# contact is at ~1.95 nm, and a 0.2 nm bead-center separation is not
# geometrically reachable. r_min is set to the bead contact distance as a
# placeholder - it should be replaced with a real bead-level PMF depth once
# one exists; the depth itself (80 kJ/mol) is also carried over from the
# atomistic value and provisional pending that recalculation.
pmf_r_min = STREP_SUBUNIT_RADIUS + BIOTIN_RADIUS  # nm, bead contact distance
pmf_depth = 80  # kJ/mol, placeholder well depth (see note above)
sigma_patch = pmf_r_min / 2 ** (1 / 6)
pair_params_cons = {'epsilon': pmf_depth, 'sigma': sigma_patch}
patches.params[('Strep_cons', 'Biotin')] = dict(pair_params=pair_params_cons,
                                 envelope_params=envelope_params_cons)
# r_cut should be set by where the LJ tail has decayed to ~0, not by where
# the PMF calculation happened to stop - the sharp feature near 3.4 nm in
# the PMF is an artifact of the finite sampling range, not a real feature
# of the interaction, so it shouldn't drive this choice. 2.5*sigma is the
# standard LJ cutoff (tail already negligible there); with the corrected,
# larger sigma_patch above, 3*sigma would exceed L/2 and violate the
# neighbor list's minimum-image convention for this box.
patches.r_cut[('Strep_cons','Biotin')] = 2.5 * sigma_patch
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
# Bead diameter = 2*BIOTIN_RADIUS, consistent with the radius used for its
# rotational inertia and Strep_cons-Biotin sterics above.
sigma_biotin = 2 * BIOTIN_RADIUS
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
# Uses the SAME sigma as the patch's own repulsive core (sigma_patch above)
# instead of a separate, smaller placeholder - so outside the cone contact
# starts at exactly the same distance PatchyLJ's repulsive branch already
# uses inside the cone, and the two potentials never disagree about where
# the beads touch.
wca = hoomd.md.pair.LJ(nlist=cell, default_r_cut=0.0, mode="shift")
wca.params.default = dict(epsilon=0, sigma=1)
epsilon_excl = 1  # kJ/mol scale, matches the PatchyLJ energy convention
wca.params[('Strep_cons', 'Biotin')] = dict(epsilon=epsilon_excl, sigma=sigma_patch)
wca.r_cut[('Strep_cons', 'Biotin')] = sigma_patch * 2 ** (1 / 6)

# Strep_cons-Strep_cons: sized to the actual subunit radius
# (STREP_SUBUNIT_RADIUS, defined above for the rigid-body inertia calc)
# instead of reusing the tiny Strep_cons-Biotin placeholder - two
# streptavidin subunits should not be able to interpenetrate the way the
# shared placeholder used to allow.
# r_min is set to the sum of the two bead radii (2 * STREP_SUBUNIT_RADIUS),
# i.e. where the spheres just touch.
sigma_strep = 2 * STREP_SUBUNIT_RADIUS / 2 ** (1 / 6)
wca.params[('Strep_cons', 'Strep_cons')] = dict(epsilon=epsilon_excl, sigma=sigma_strep)
wca.r_cut[('Strep_cons', 'Strep_cons')] = sigma_strep * 2 ** (1 / 6)

# Wall representing the Eppendorf tube's inner surface (virgin
# polypropylene - hydrophobic, untreated plastic). Biotin: plain volume
# exclusion is a reasonable approximation - no strong documented plastic-
# adsorption tendency for small, mostly-polar free biotin. Strep_cons:
# pure exclusion is NOT accurate - streptavidin's hydrophobic surface
# patches are known to nonspecifically physisorb onto untreated
# polypropylene (this is literally how streptavidin-coated PP plates are
# made, by passive adsorption), so its wall potential keeps an attractive
# tail instead of being truncated at the repulsive core like Biotin's.
#
# TUBE_RADIUS/TUBE_HALF_HEIGHT defined up top (reused below for sampling
# biotin's starting positions). End-cap planes at +-TUBE_HALF_HEIGHT -
# previously left at +-10, outside the box entirely (it only spans z in
# [-6.25,6.25] for L=12.5), so they never actually capped anything.
cylindrical_wall = hoomd.wall.Cylinder(radius=TUBE_RADIUS, axis=(0, 0, 1))
top = hoomd.wall.Plane(origin=(0, 0, TUBE_HALF_HEIGHT), normal=(0, 0, -1))
bottom = hoomd.wall.Plane(origin=(0, 0, -TUBE_HALF_HEIGHT), normal=(0, 0, 1))
wallLJ = hoomd.md.external.wall.ForceShiftedLJ([cylindrical_wall, top, bottom])
wallLJ.params.default = dict(epsilon=0, sigma=1, r_cut=0)

# Biotin: WCA-style pure repulsion. sigma = R/2**(1/6) (same pattern as
# sigma_patch/sigma_strep above) so the potential's own minimum - and
# therefore r_cut, truncated right there - lands exactly at r =
# BIOTIN_RADIUS from the wall surface, i.e. where the bead's edge touches
# the wall. epsilon reuses epsilon_excl, the same steric-strength
# convention already used for the bulk WCA terms.
sigma_wall_biotin = BIOTIN_RADIUS / 2 ** (1 / 6)
wallLJ.params["Biotin"] = dict(
    epsilon=epsilon_excl,
    sigma=sigma_wall_biotin,
    r_cut=sigma_wall_biotin * 2 ** (1 / 6),  # = BIOTIN_RADIUS, repulsive-only
)

# Strep_cons: same repulsive-core construction (sigma tied to
# STREP_SUBUNIT_RADIUS instead), but r_cut extended past the minimum
# (2.5*sigma, matching the Strep_cons-Biotin patch cutoff convention
# above) so the attractive LJ tail survives - representing nonspecific
# hydrophobic physisorption to the polypropylene wall.
#
# No literature value exists for streptavidin-on-virgin-polypropylene
# specifically. Anchored instead to the closest comparable number found
# (an MD estimate of ~-29 kJ/mol for the strongest generic protein-
# adsorption orientation on a hydrophobic surface) rather than picked
# arbitrarily - still a placeholder pending real calibration, and
# deliberately kept well below the specific biotin-pocket well (80 kJ/mol)
# so nonspecific wall sticking doesn't rival the designed binding
# interaction.
WALL_EPSILON_STREP = 29  # kJ/mol, placeholder (see note above)
sigma_wall_strep = STREP_SUBUNIT_RADIUS / 2 ** (1 / 6)
wallLJ.params["Strep_cons"] = dict(
    epsilon=WALL_EPSILON_STREP,
    sigma=sigma_wall_strep,
    r_cut=2.5 * sigma_wall_strep,
)

integrator.forces.append(patches)
integrator.forces.append(LJ)
integrator.forces.append(wca)
integrator.forces.append(wallLJ)



rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.Langevin(
    filter=rigid_centers_and_free, kT=kT
)

# Per-type friction from Stokes-Einstein drag in an implicit solvent,
# instead of the HOOMD default (gamma=1.0 for every type regardless of
# size). That default made translational relaxation time (mass/gamma)
# differ between Strep_cent and Biotin only because their masses differ -
# not because of any real difference in solvent drag - which is what made
# the tetramer diffuse like a free particle instead of barely moving.
# gamma_trans = 6*pi*eta*r, gamma_rot = 8*pi*eta*r^3 (SI Stokes drag).
# 1 Pa*s = 1 kg/(m*s); converting kg->g/mol via Avogadro's number and
# m->nm, s->ps gives the factor below - exact, given the length=nm,
# mass=g/mol, energy=kJ/mol convention already used throughout this file.
AVOGADRO = 6.02214076e23  # 1/mol
PAS_TO_SIM_VISCOSITY = AVOGADRO * 1e3 * 1e-9 * 1e-12  # Pa*s -> g/mol/(nm*ps)
SOLVENT_VISCOSITY_PAS = 8.9e-4  # water at 25 C; ~6.9e-4 for 37 C if modeling physiological temp
eta = SOLVENT_VISCOSITY_PAS * PAS_TO_SIM_VISCOSITY

# Effective hydrodynamic radius of the whole rigid tetramer: a subunit's
# own radius plus how far that subunit sits from the body center.
STREP_BODY_RADIUS = STREP_SUBUNIT_RADIUS + numpy.linalg.norm(initial_cons_pos[0])

langevin.gamma["Strep_cent"] = 6 * math.pi * eta * STREP_BODY_RADIUS
langevin.gamma["Biotin"] = 6 * math.pi * eta * BIOTIN_RADIUS
langevin.gamma_r["Strep_cent"] = (8 * math.pi * eta * STREP_BODY_RADIUS**3,) * 3
langevin.gamma_r["Biotin"] = (8 * math.pi * eta * BIOTIN_RADIUS**3,) * 3

integrator.methods.append(langevin)

simulation.operations.integrator = integrator

# periodically write frames to a trajectory over the whole run, instead of
# only writing the final state once the run finishes.
# Trigger period scaled up along with nsteps below (100 -> 5000) to keep the
# frame count (and file size) in the same ballpark as before, rather than
# writing 15,000 frames for a 150x longer run.
trajectory_writer = hoomd.write.GSD(
    filename="trajectory.gsd",
    trigger=hoomd.trigger.Periodic(5000),
    mode="wb",
)
simulation.operations.writers.append(trajectory_writer)

simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

# 1.5M steps * dt=0.002 ps = 3 ns - enough for Strep_cent to show a visible
# (~1nm) diffusive wander (tau_trans ~1.5ps, so this is ~2000 relaxation
# times), at ~6800 steps/sec on CPU that's only ~4 min of wall time. Full
# tetramer rotation would need ~80M steps (tau_rot ~0.3ps but D_rot is tiny
# for this body size) - not attempted here, that's a separate, much longer
# run if ever needed.
simulation.run(1500000)
simulation.operations.writers.remove(trajectory_writer)
del trajectory_writer

#traj = gsd.hoomd.open("trajectory.gsd")
#rdr.render_movie(traj)
#can ad npt if needed
#npt = hoomd
