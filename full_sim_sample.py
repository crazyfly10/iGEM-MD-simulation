# each Het-S spans 9.5 angstronms. In the paper
# each beta amyloid spans 4.7 angtroms, every 3 beta amyloids
# are approximated as a bead, so every 1.5 Het-S would be approximated
# as a bead. so each bead is approx 1.4 nm in diameter.
# thats around 5000/9.5 = 526 Het-S. if we want every 1.5 het-s as a bead
# 526/1.5 is approx 351 beads

# assume a malemide bead with diameter 0.7, so radius of 0.35
# biotin 0.4 nm radius
# strep monomer 1.7 nm radius
# peg repeating 0.17 nm radius
# 1.4 nm in diamter for het-s so 0.7 nm radius

import math
import datetime
#import pandas as pd
import gsd.hoomd
import hoomd
import numpy as np

def fibril_relative_pos(N_HETs, rep_unit):
    # N_HETs:   number of Het-s beads in the fibril backbone.
    # rep_unit: number of PEG repeating units in EACH biotin-PEG-malemide
    #           (one per fibril end). The malemide-adjacent PEG bead is the
    #           first of these, so the zig-zag loops below add rep_unit - 1
    #           more beads per side.
    Hets_radius = 0.7 #nm
    Het_s_pos = np.empty((0,3))
    # align along y axis, centred on the origin. Offsetting by (N_HETs-1)/2
    # rather than using a symmetric range() keeps the count exactly N_HETs
    # for even and odd N_HETs alike.
    for i in range(N_HETs):
        Het_s_pos = np.append(Het_s_pos, [[0,2*Hets_radius*(i - (N_HETs-1)/2),0]], axis = 0)

    #add malemide
    # assume PEG-Malemide-HETs angle is 120
    mal_radius = 0.35

    P_M_H_ang = math.pi * (2/3)
    z_mal = (mal_radius + Hets_radius) * math.sin(math.pi/4 - P_M_H_ang/2)
    y_mal = (mal_radius + Hets_radius) * math.cos(math.pi/4 - P_M_H_ang/2)

    # last_mal uses -z_mal, not +z_mal - NOT a typo. The rest of this
    # function builds a proper zigzag by writing the SAME literal pattern for
    # a "step away from the backbone" on both arms (e.g. the PEG-PEG zigzag
    # below writes "-z_peg_d" for the i=0 step on BOTH left_pegs and
    # right_pegs - not mirrored - only y mirrors between arms, z follows the
    # same alternation). first_mal (the left arm's i=0 step) already follows
    # that convention (-z_mal). last_mal (the right arm's own i=0 step)
    # previously used +z_mal - a different sign for the same step index,
    # breaking the arms' mirror symmetry. This one sign was the actual root
    # cause of the asymmetry/degeneracy; a compensating sign flip elsewhere
    # (tried and rejected: flipping Mal_peg_r) only traded one broken
    # symmetry for another; only fixing the true source (this line) makes
    # PEG-PEG-Malemide, PEG-Malemide-HETS AND Malemide-HETS-HETS all
    # symmetric between the two arms simultaneously, at every rep_unit and
    # N_HETs tried (verified computationally, not just at this file's own
    # values).
    first_mal = np.array([Het_s_pos[0]]) + np.array([0,-y_mal,-z_mal])
    last_mal = np.array([Het_s_pos[-1]]) + np.array([0,y_mal,-z_mal])
    Mal_w_pri = np.vstack([
        first_mal,
        Het_s_pos,
        last_mal,
    ])

    # add PEG-mal
    peg_radius = 0.17
    z_fpeg = (mal_radius + peg_radius) * math.sin(math.pi/4 - P_M_H_ang/2)
    y_fpeg = (mal_radius + peg_radius) * math.cos(math.pi/4 - P_M_H_ang/2)
    # Mal_peg_l/Mal_peg_r ARE meant to share the same literal sign (+z_fpeg,
    # the i=1 step of the same convention described above) - this was
    # already correct before the last_mal fix above; it just couldn't
    # produce symmetric angles while last_mal was still wrong.
    Mal_peg_l = np.array(Mal_w_pri[0]) + np.array([0,-y_fpeg,z_fpeg])
    Mal_peg_r = np.array(Mal_w_pri[-1]) + np.array([0,y_fpeg,z_fpeg])
    P1_M_H_chain = np.vstack([
        Mal_peg_l,
        Mal_w_pri,
        Mal_peg_r,
    ])

    # add the rest of the PEGs
    peg_peg_ang = math.radians(110)
    left_pegs = np.array([P1_M_H_chain[0].copy()])
    right_pegs = np.array([P1_M_H_chain[-1].copy()])
    y_peg_d = 2 * peg_radius * math.cos(math.pi/4 - peg_peg_ang/2)
    z_peg_d = 2 * peg_radius * math.sin(math.pi/4 - peg_peg_ang/2)
    for i in range(rep_unit-1):
        if i % 2 == 0:
            v = np.array([0,-y_peg_d,-z_peg_d])
        else:
            v = np.array([0,-y_peg_d,z_peg_d])
        current = left_pegs[i].copy() + v
        left_pegs = np.vstack([
            left_pegs,
            current
        ])
    for i in range(rep_unit-1):
        if i % 2 == 0:
            v = np.array([0, y_peg_d,-z_peg_d])
        else:
            v = np.array([0, y_peg_d, z_peg_d])
        current = right_pegs[i].copy() + v
        right_pegs = np.vstack([
            right_pegs,
            current
        ])
    left_pegs = np.delete(left_pegs,0,0)
    right_pegs = np.delete(right_pegs,0,0)
    left_pegs = np.flip(left_pegs,0)
    full_peg_M_H = np.vstack([
        left_pegs,
        P1_M_H_chain,
        right_pegs,
    ])

    # add biotin
    biotin_radius = 0.4
    biotin_peg_ang = math.radians(110)
    B_y_d = (biotin_radius + peg_radius) * math.cos(math.pi/4 - biotin_peg_ang/2)
    B_z_d = (biotin_radius + peg_radius) * math.sin(math.pi/4 - biotin_peg_ang/2)
    # A fixed -B_z_d here (as this line previously had, on both ends) only
    # continues the zigzag's existing alternation correctly for HALF of the
    # possible rep_unit values - whichever z-direction the outermost PEG bead
    # already ended up facing depends on the PARITY of rep_unit (each zigzag
    # step flips z, so after rep_unit-1 steps the sign has flipped that many
    # times). At odd rep_unit -B_z_d continues the zigzag correctly; at even
    # rep_unit (this file's actual REP_UNIT_PER_ARM=68) it instead lands
    # exactly in line with the previous two beads, making the biotin
    # collinear with them - a degenerate Biotin-PEG-PEG-PEG dihedral, and
    # confirmed to happen at every even rep_unit tried (4, 6, 8, 68). Both
    # ends need the SAME sign (verified directly, not assumed by symmetry),
    # parity-dependent via (-1)**rep_unit, so this is correct for any
    # rep_unit rather than only the one currently hardcoded in the file.
    biotin_z_sign = (-1) ** rep_unit
    l_biotin = np.array(full_peg_M_H[0]) + np.array([0,-B_y_d, biotin_z_sign * B_z_d])
    r_biotin = np.array(full_peg_M_H[-1]) + np.array([0,B_y_d, biotin_z_sign * B_z_d])
    full_fib = np.vstack([
        l_biotin,
        full_peg_M_H,
        r_biotin
    ])

    # per-bead typeid, built in the same order as the vstack above so it
    # stays in sync automatically if rep_unit/N_HETs change. Mapping:
    # Biotin=0, PEG=1, Malemide=2, HETS=3 (Strep_cent=4/Strep_cons=5 are
    # assigned elsewhere, for the streptavidin cores).
    fib_typeid = (
        [0]                        # l_biotin
        + [1] * (rep_unit - 1)     # left_pegs
        + [1]                      # Mal_peg_l
        + [2]                      # first_mal
        + [3] * N_HETs             # Het_s_pos
        + [2]                      # last_mal
        + [1]                      # Mal_peg_r
        + [1] * (rep_unit - 1)     # right_pegs
        + [0]                      # r_biotin
    )

    return full_fib.shape[0], full_fib, fib_typeid

initial_cons_pos = np.array(
    [[-1.202,-1.202,-1.202],
     [1.202,1.202,-1.202 ],
     [-1.202,1.202,1.202],
     [1.202,-1.202,1.202]]
)

# ---------------------------------------------------------------------------
# State point: [fibril] = 0.5 uM.
#
# This is set FIRST and deliberately, because the MS-IBI potentials that will
# replace the interim force parameters below are derived at a state point -
# the all-atom reference states should bracket the one simulated here.
#
# (A 2.0uM alternative was tested and measured: raising concentration is the
# strongest, and only genuinely free, lever on percolation odds within a
# fixed particle/runtime budget - distance-to-close scales as [fibril]^(-1/3)
# - and it costs nothing since N_FIBRIL/N_STREP stay fixed, only the cylinder
# shrinks. It raised the fraction of biotin ends within diffusive reach of a
# site from 18.8% to 41.8%, but the resulting connected cluster only reached
# ~5 of 600 Strep_cent nodes - a real but incremental effect, not a fix - so
# 0.5uM was kept.)
#
# Sizing is on the SOLUTION volume (the cylindrical wall's interior, where the
# fibrils and streptavidin actually are), not the cube:
#     V_cyl = pi R^2 H = N_FIBRIL / (N_A * 0.5e-6 mol/L) = 6.64e8 nm^3
# With an H = 2R aspect that gives R = 473 nm, H = 946 nm.
#
# The box is then L = 2 * (R + margin). HOOMD boxes are ALWAYS periodic -
# there is no "PBC off" switch - so "no periodic boundaries" is achieved by
# keeping every particle at least one interaction cutoff away from the box
# face, which the wall does. The 20nm margin comfortably exceeds the largest
# r_cut in the system (7.6nm, the Strep_cons-Strep_cons attractive tail), so
# the periodic images are never reachable. That, not box size alone, is what
# removes the fibril self-image problem: a fibril is up to ~584nm long, far
# more than L/2, so this model would be unusable with real PBC.
#
# fibril_relative_pos() itself never references the cylinder - it only builds
# a fibril's LOCAL bead template, centred on the origin. The cylinder flows
# entirely through these two constants -> L -> randomise_positions() ->
# _sample_cylinder_center() -> the wall geometry below, so changing them here
# is sufficient; nothing in fibril_relative_pos needs to change.
CYL_RADIUS = 473.0        # nm, cylindrical wall (Eppendorf interior) - the
CYL_HALF_HEIGHT = 473.0   # nm, wall itself is a separate follow-up
WALL_BOX_MARGIN = 20.0    # nm, > max r_cut so no particle ever sees an image
L = 2 * (CYL_RADIUS + WALL_BOX_MARGIN)  # 986 nm

# ---------------------------------------------------------------------------
# randomised placement of fibrils and streptavidin cores in the cylinder
# ---------------------------------------------------------------------------
# N chosen to hit 0.5 uM while keeping the run inside a ~2 day budget on an
# RTX 5090: ~99,600 particles, scaling to roughly 0.9-2.3 days for the 150M
# step run. The Strep:fibril ratio (currently 3:1) was checked directly
# against 2:1 and 4:1 (at a higher, since-reverted test concentration) and
# found not to matter - the resulting connected cluster size changes by well
# under the run-to-run noise - so it was left alone rather than retuned.
N_FIBRIL = 200
N_STREP = 600

# Single source of truth for every random draw in this file (fibril-length
# distribution, randomise_positions' placement/orientation rng, and HOOMD's
# own seed for thermalize_particle_momenta + Langevin noise) - previously
# three separate hardcoded 0s. Change this one value to get an independent
# replica; everything else in the file derives from it.
SIMULATION_SEED = 1

REP_UNIT_PER_ARM = 68  # PEG repeat units per arm - same for every fibril

# particles.typeid <-> particles.types mapping, index-for-index
TYPES = ["Biotin", "PEG", "Malemide", "HETS", "Strep_cent", "Strep_cons"]
BIOTIN, PEG, MALEMIDE, HETS, STREP_CENT, STREP_CONS = range(6)

# ---------------------------------------------------------------------------
# per-type bead mass (g/mol), built from the atoms each bead represents
# ---------------------------------------------------------------------------
# Standard atomic weights (g/mol)
C, H, N, O = 12.011, 1.008, 14.007, 15.999

# -C(=O)-NH- : the amide linkage joining biotin/maleimide onto the PEG arm.
AMIDE_GROUP_MASS = C + O + N + H  # 43.03

BIOTIN_MASS = 244.31  # g/mol, free biotin (C10H16N2O3S) - matches
                       # sample_strep-biotin_sim_fixed.py's BIOTIN_MASS
BIOTIN_BEAD_MASS = BIOTIN_MASS + AMIDE_GROUP_MASS  # biotin + its amide linker

EO_UNIT_MASS = 2*C + 4*H + O  # 44.05, one -CH2-CH2-O- ethylene-oxide repeat

MALEIMIDE_MASS = 4*C + 3*H + N + 2*O  # 97.07, free maleimide (C4H3NO2)
MALEIMIDE_BEAD_MASS = MALEIMIDE_MASS + AMIDE_GROUP_MASS  # + its amide linker

# PLACEHOLDER: one "Het-S" unit (per this file's own header comment) is
# taken as the HET-s(218-289) prion-forming domain, 72 residues, at the
# standard ~110 Da/residue average peptide-bond mass, plus one water's worth
# for the two free termini - not a measured value, replace if a real one
# becomes available.
HETS_MONOMER_MASS = 72 * 110 + 18  # ~7938 g/mol
HETS_BEAD_MASS = 1.5 * HETS_MONOMER_MASS  # a bead is 1.5 Het-S units

# Reuses sample_strep-biotin_sim_fixed.py's per-subunit estimate
# (~13-16 kDa tetramer subunit). Strep_cent carries the mass of the WHOLE
# rigid body (HOOMD convention: the central particle, not a per-particle
# placeholder) since its four Strep_cons constituents aren't placed yet.
STREP_SUBUNIT_MASS = 14500  # g/mol, per subunit
STREP_CENT_MASS = 4 * STREP_SUBUNIT_MASS

MASS_BY_TYPEID = {
    BIOTIN: BIOTIN_BEAD_MASS,
    PEG: EO_UNIT_MASS,
    MALEMIDE: MALEIMIDE_BEAD_MASS,
    HETS: HETS_BEAD_MASS,
    STREP_CENT: STREP_CENT_MASS,
    # STREP_CONS is never assigned here - its constituents aren't placed in
    # this snapshot, so there's no mass[i] that would need this value.
}

STREP_SUBUNIT_RADIUS = 1.7 #nm

# Bead radii (nm) - duplicated here at module scope from fibril_relative_pos's
# local variables of the same values, so the pair-potential section further
# below can build WCA sigmas from them without hardcoding disconnected
# numbers.
BIOTIN_RADIUS = 0.4    # matches fibril_relative_pos's biotin_radius. Note
                        # this differs from sample_strep-biotin_sim_fixed.py's
                        # own BIOTIN_RADIUS=0.25 (that file models a free-
                        # floating biotin molecule; here Biotin is always
                        # this fibril's own bead), so PMF-derived values
                        # below use THIS radius, not that file's, to stay
                        # internally consistent - reusing that file's PMF
                        # METHOD (depth, envelope shape), not its numbers.
PEG_RADIUS = 0.17
MALEMIDE_RADIUS = 0.35
HETS_RADIUS = 0.7

RADIUS_BY_TYPEID = {
    BIOTIN: BIOTIN_RADIUS,
    PEG: PEG_RADIUS,
    MALEMIDE: MALEMIDE_RADIUS,
    HETS: HETS_RADIUS,
    STREP_CONS: STREP_SUBUNIT_RADIUS,
    # STREP_CENT excluded - it's non-interacting for every pair (see the
    # pair-potential section further below), so it never needs a radius here.
}

# Biotin-Biotin gets a LARGER exclusion radius than BIOTIN_RADIUS, and only
# for that one pair. This is what actually enforces "one biotin per
# streptavidin monomer" - the reason alpha=pi/12 was chosen for the PatchyLJ
# envelope below, which on its own does NOT achieve it:
#
# Biotin binds at r = STREP_SUBUNIT_RADIUS + BIOTIN_RADIUS = 2.1nm from the
# Strep_cons centre. Two biotins only need to clear each other sterically, so
# at the default 0.4nm radius (contact 0.8nm) a second biotin can sit just
# asin(0.8/(2*2.1)) = 11 degrees off the patch axis - INSIDE the 15 degree
# cone, where it still receives 84% of the envelope, so it binds too.
# Narrowing alpha barely helps (at alpha=8 deg it needs omega>=300, and costs
# another ~12x in capture rate, which this system cannot afford).
#
# Raising the Biotin-Biotin contact distance to 1.1nm pushes that second
# biotin out to 15.2 degrees - outside the cone - enforcing single occupancy
# sterically at zero cost to the binding rate. It also mirrors the real
# mechanism: a pocket that physically accommodates exactly one biotin.
#
# Deliberately NOT done by changing BIOTIN_RADIUS, which would also perturb
# biotin's mass-independent inertia, its binding distance to Strep_cons, and
# every other biotin pair.
BIOTIN_EXCL_RADIUS = 0.55  # nm, Biotin-Biotin steric only (see above)


def wca_sigma(typeid_a, typeid_b):
    if typeid_a == BIOTIN and typeid_b == BIOTIN:
        return 2 * BIOTIN_EXCL_RADIUS / 2 ** (1 / 6)
    return (RADIUS_BY_TYPEID[typeid_a] + RADIUS_BY_TYPEID[typeid_b]) / 2 ** (1 / 6)


# Only the Strep_cent core particle is placed here - the rigid body is not
# defined yet, so its four Strep_cons constituents do not exist in the
# snapshot and will be built later by
# hoomd.md.constrain.Rigid().create_bodies(). Their space still has to be
# reserved now, otherwise a constituent gets created sitting on top of a
# fibril bead. A sphere of this radius about the core encloses every
# constituent at ANY orientation, so the test holds however the body ends up
# being oriented.
STREP_EXCL_RADIUS = np.linalg.norm(initial_cons_pos[0]) + STREP_SUBUNIT_RADIUS

# Placement-only clearance about a fibril backbone. The fattest fibril bead is
# Het-s at 0.7 nm radius; the remainder is margin so nothing starts inside a
# neighbour's repulsive core.
FIBRIL_EXCL_RADIUS = 1.0 #nm

# ---------------------------------------------------------------------------
# per-type moment of inertia. Same self-inertia + parallel-axis-theorem +
# diagonalisation steps already used to build a rigid body's central-particle
# moment of inertia in sample_strep-biotin_sim_fixed.py and
# Biotin_PEG_Sterp-sample_sim.py, factored out into reusable functions here.
# ---------------------------------------------------------------------------
def solid_sphere_inertia(mass, radius):
    # self-inertia of a single solid-sphere constituent about its own centre
    return np.identity(3) * (2 / 5 * mass * radius**2)


def rigid_body_moment_inertia(constituent_mass, constituent_radius, constituent_positions):
    # Sums each constituent's own self-inertia, shifted out to its offset
    # from the body centre via the parallel axis theorem, then diagonalises
    # the resulting tensor to get the body's principal moments (I_diagonal) -
    # what particles.moment_inertia actually wants, since HOOMD expects it
    # expressed in the body's own principal-axis frame, with
    # particles.orientation carrying the rotation from that frame to world.
    #
    # Diagonalising also gives the eigenvectors (E_vec) - the rotation that
    # takes the ORIGINAL constituent_positions into that same principal-axis
    # frame. constituent_positions as originally given aren't necessarily
    # already expressed in it, so this rotation has to be applied to them
    # too (R = E_vec.T, new_positions = R @ constituent_positions.T) before
    # they're used anywhere alongside I_diagonal (e.g. as
    # rigid.body[...]["positions"]) - otherwise the body's assumed mass
    # distribution (I_diagonal) and its actual constituent placement would
    # be in two different, mismatched frames.
    I_self = solid_sphere_inertia(constituent_mass, constituent_radius)
    I_general = np.zeros((3, 3))
    for r in constituent_positions:
        I_general += I_self + constituent_mass * (
            np.dot(r, r) * np.identity(3) - np.outer(r, r)
        )
    I_diagonal, E_vec = np.linalg.eig(I_general)
    I_diagonal = np.real(I_diagonal)
    E_vec = np.real(E_vec)
    R = E_vec.T
    new_positions = np.dot(R, np.asarray(constituent_positions).T).T
    return I_diagonal, new_positions


# Strep_cent's principal moments, and its four Strep_cons constituents'
# positions re-expressed in that same principal-axis frame (see function
# comment above) - used later for rigid.body["Strep_cent"]["positions"].
STREP_CENT_MOMENT_INERTIA, STREP_CONS_POS = rigid_body_moment_inertia(
    STREP_SUBUNIT_MASS, STREP_SUBUNIT_RADIUS, initial_cons_pos
)
STREP_CENT_MOMENT_INERTIA = tuple(STREP_CENT_MOMENT_INERTIA)

# Biotin is now a patchy particle (patches.directors["Biotin"] in the
# PatchyLJ section below), so unlike the other fibril bead types it DOES need
# a real rotational degree of freedom - zero moment_inertia would leave its
# director frozen at its initial orientation forever, since HOOMD skips
# rotational integration for particles with zero moment of inertia, so no
# torque from the patchy potential (or from Langevin's rotational noise)
# could ever turn it. Treated as an isotropic solid sphere of its own bead
# mass/radius (same solid_sphere_inertia() helper used for Strep_cent above,
# same fix sample_strep-biotin_sim_fixed.py made for its own free Biotin -
# see the comment there: reusing Strep's mass/radius by mistake there gave
# Biotin ~2700x too much rotational inertia to visibly rotate over a short
# run).
BIOTIN_MOMENT_INERTIA = tuple(np.diagonal(solid_sphere_inertia(BIOTIN_BEAD_MASS, BIOTIN_RADIUS)))

MOMENT_INERTIA_BY_TYPEID = {
    # PEG/Malemide/HETS fibril beads still have no anisotropic potential, so
    # their own orientation stays physically inert - isotropic/zero
    # moment_inertia is what makes that true, matching Biotin_PEG_Sterp-
    # sample_sim.py's (0,0,0) convention for its own non-patchy free beads.
    BIOTIN: BIOTIN_MOMENT_INERTIA,
    PEG: (0.0, 0.0, 0.0),
    MALEMIDE: (0.0, 0.0, 0.0),
    HETS: (0.0, 0.0, 0.0),
    STREP_CENT: STREP_CENT_MOMENT_INERTIA,
    # STREP_CONS is never assigned here - its constituents aren't placed in
    # this snapshot, same reasoning as MASS_BY_TYPEID above.
}

# ---------------------------------------------------------------------------
# bond types. Bonds only ever connect consecutive beads within one fibril's
# backbone - streptavidin cores have no bonds at all. Named after the pair of
# particle types each one connects, same convention
# Biotin_PEG_Sterp-sample_sim.py uses.
# ---------------------------------------------------------------------------
BOND_TYPES = ["Biotin-PEG", "PEG-PEG", "PEG-Malemide", "Malemide-HETS", "HETS-HETS"]
BOND_TYPE_INDEX = {name: i for i, name in enumerate(BOND_TYPES)}

def _bond_key(typeid_a, typeid_b):
    return tuple(sorted((typeid_a, typeid_b)))

# Every consecutive-bead pair fibril_relative_pos can produce, regardless of
# N_HETs/rep_unit (verified against its bead ordering: Biotin, PEG..., PEG,
# Malemide, HETS..., Malemide, PEG, PEG..., Biotin) maps to one of these -
# order doesn't matter since _bond_key sorts it.
BOND_TYPE_BY_PARTICLE_TYPEID_PAIR = {
    _bond_key(BIOTIN, PEG): BOND_TYPE_INDEX["Biotin-PEG"],
    _bond_key(PEG, PEG): BOND_TYPE_INDEX["PEG-PEG"],
    _bond_key(PEG, MALEMIDE): BOND_TYPE_INDEX["PEG-Malemide"],
    _bond_key(MALEMIDE, HETS): BOND_TYPE_INDEX["Malemide-HETS"],
    _bond_key(HETS, HETS): BOND_TYPE_INDEX["HETS-HETS"],
}

# ---------------------------------------------------------------------------
# angle types. Every 3 consecutive beads within one fibril's backbone form an
# angle - unlike dihedrals, a bond angle has no collinearity degeneracy (3
# exactly-collinear Het-s beads just give a well-defined 180 degree angle,
# not an undefined one), so no windows need to be excluded here. Streptavidin
# cores have no angles at all.
# ---------------------------------------------------------------------------
ANGLE_TYPES = [
    "Biotin-PEG-PEG",
    "PEG-PEG-PEG",
    "PEG-PEG-Malemide",
    "PEG-Malemide-HETS",
    "Malemide-HETS-HETS",
    "HETS-HETS-HETS",
]
ANGLE_TYPE_INDEX = {name: i for i, name in enumerate(ANGLE_TYPES)}

def _angle_key(typeid_a, typeid_b, typeid_c):
    window = (typeid_a, typeid_b, typeid_c)
    return min(window, tuple(reversed(window)))

# Every consecutive-3-bead window fibril_relative_pos can produce maps to one
# of these - order doesn't matter since _angle_key takes the
# lexicographically-smaller of the window and its reverse (reading an angle
# forward or backward is the same physical angle).
ANGLE_TYPE_BY_PARTICLE_TYPEID_WINDOW = {
    _angle_key(BIOTIN, PEG, PEG): ANGLE_TYPE_INDEX["Biotin-PEG-PEG"],
    _angle_key(PEG, PEG, PEG): ANGLE_TYPE_INDEX["PEG-PEG-PEG"],
    _angle_key(PEG, PEG, MALEMIDE): ANGLE_TYPE_INDEX["PEG-PEG-Malemide"],
    _angle_key(PEG, MALEMIDE, HETS): ANGLE_TYPE_INDEX["PEG-Malemide-HETS"],
    _angle_key(MALEMIDE, HETS, HETS): ANGLE_TYPE_INDEX["Malemide-HETS-HETS"],
    _angle_key(HETS, HETS, HETS): ANGLE_TYPE_INDEX["HETS-HETS-HETS"],
}


# ---------------------------------------------------------------------------
# dihedral types. Every 4 consecutive beads within one fibril's backbone form
# a dihedral - EXCEPT where 3 of those 4 (or all 4) are Het-s: Het_s_pos
# places every Het-s bead exactly on the backbone axis (x=0, z=0), so 3
# consecutive Het-s beads are always exactly collinear, making the torsion
# angle undefined there (the dihedral formula's two bond-vector cross
# products degenerate to zero). PEG's zig-zag isn't collinear, so
# PEG-PEG-PEG-PEG stays a valid, defined dihedral. Streptavidin cores have no
# dihedrals at all.
# ---------------------------------------------------------------------------
DIHEDRAL_TYPES = [
    "Biotin-PEG-PEG-PEG",
    "PEG-PEG-PEG-PEG",
    "PEG-PEG-PEG-Malemide",
    "PEG-PEG-Malemide-HETS",
    "PEG-Malemide-HETS-HETS",
]
DIHEDRAL_TYPE_INDEX = {name: i for i, name in enumerate(DIHEDRAL_TYPES)}

def _dihedral_key(typeid_a, typeid_b, typeid_c, typeid_d):
    window = (typeid_a, typeid_b, typeid_c, typeid_d)
    return min(window, tuple(reversed(window)))

# Every non-degenerate consecutive-4-bead window fibril_relative_pos can
# produce maps to one of these - order doesn't matter since _dihedral_key
# takes the lexicographically-smaller of the window and its reverse (reading
# a dihedral forward or backward is the same physical torsion).
DIHEDRAL_TYPE_BY_PARTICLE_TYPEID_WINDOW = {
    _dihedral_key(BIOTIN, PEG, PEG, PEG): DIHEDRAL_TYPE_INDEX["Biotin-PEG-PEG-PEG"],
    _dihedral_key(PEG, PEG, PEG, PEG): DIHEDRAL_TYPE_INDEX["PEG-PEG-PEG-PEG"],
    _dihedral_key(PEG, PEG, PEG, MALEMIDE): DIHEDRAL_TYPE_INDEX["PEG-PEG-PEG-Malemide"],
    _dihedral_key(PEG, PEG, MALEMIDE, HETS): DIHEDRAL_TYPE_INDEX["PEG-PEG-Malemide-HETS"],
    _dihedral_key(PEG, MALEMIDE, HETS, HETS): DIHEDRAL_TYPE_INDEX["PEG-Malemide-HETS-HETS"],
}

def _is_degenerate_dihedral_window(a, b, c, d):
    # 3 consecutive Het-s beads (either the first or last 3 of the window)
    # are exactly collinear - see section comment above.
    return (a == HETS and b == HETS and c == HETS) or (b == HETS and c == HETS and d == HETS)


def random_unit_quaternion(rng):
    # Shoemake's method - uniform over all 3D rotations. Scalar-first
    # (w,x,y,z), the convention HOOMD uses.
    u1, u2, u3 = rng.random(3)
    return np.array([
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
    ])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


# For the overlap tests a fibril is treated as a capsule running between its
# two end beads. The backbone's internal zig-zag is only ~1 nm wide against a
# ~500 nm length, so a line segment is a faithful stand-in and keeps the test
# cheap enough to run on every insertion attempt.
def point_segment_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    t = 0.0 if denom == 0 else np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def segment_segment_dist(p1, q1, p2, q2, eps=1e-10):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    if a <= eps and e <= eps:
        return np.linalg.norm(p1 - p2)
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom != 0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return np.linalg.norm(c1 - c2)


def _sample_cylinder_center(rng, radius_bound, half_height_bound):
    # Uniform placement WITHIN the cylindrical wall, not the box - needed so
    # nothing gets placed in a box corner outside the wall, where it would
    # sit in enormous WCA overlap with wallLJ and be violently ejected on the
    # first integration step (confirmed empirically: ~2% of particles landed
    # outside the cylinder under plain box-uniform sampling).
    #
    # sqrt(uniform) rather than uniform(0, radius_bound) directly: points
    # uniform per unit AREA of a disk are NOT uniform in r (the area element
    # grows as r dr, so the correct radial CDF is r**2, inverted here) - same
    # method sample_strep-biotin_sim_fixed.py uses for its own cylindrical
    # tube. radius_bound/half_height_bound are the object's own placement
    # margins (already shrunk by its half-extent), analogous to fib_bound/
    # strep_bound in the box-based version this replaces.
    r = math.sqrt(rng.uniform(0, 1)) * radius_bound
    theta = rng.uniform(0, 2 * math.pi)
    z = rng.uniform(-half_height_bound, half_height_bound)
    return np.array([r * math.cos(theta), r * math.sin(theta), z])


def randomise_positions(n_hets_per_fibril, n_strep, box_L, cyl_radius, cyl_half_height,
                         seed=0, max_attempts=10000):
    # n_hets_per_fibril: sequence of Het-s bead counts, one entry per
    # fibril to place (its length sets how many fibrils get placed). Real
    # fibrils aren't all the same length, so this comes from whatever
    # length distribution the rest of the project supplies rather than
    # being generated in here.
    #
    # box_L is used only for the final sanity check below (the cylinder must
    # fit inside the box with room for the wall margin); cyl_radius/
    # cyl_half_height are the actual placement volume, matching the
    # cylindrical wall (CYL_RADIUS/CYL_HALF_HEIGHT) the state point above was
    # sized against.
    rng = np.random.default_rng(seed)

    strep_radius_bound = cyl_radius - STREP_EXCL_RADIUS
    strep_height_bound = cyl_half_height - STREP_EXCL_RADIUS
    if strep_radius_bound <= 0 or strep_height_bound <= 0:
        raise ValueError("cylinder too small to hold a streptavidin core")

    fibril_segments = []   # (p0, p1) world-space end beads of each fibril
    fibril_positions = []
    fibril_typeids = []
    fibril_bonds = []       # (i, j, bond_typeid) global particle indices
    fibril_angles = []      # (i, j, k, angle_typeid) global particle indices
    fibril_dihedrals = []   # (i, j, k, l, dihedral_typeid) global particle indices
    strep_centers = []
    strep_orientations = []

    for n_hets in n_hets_per_fibril:
        # Each fibril gets its own template built from its own supplied
        # length, rather than every fibril sharing one fixed template.
        _, fib_template, fib_typeid = fibril_relative_pos(n_hets, REP_UNIT_PER_ARM)

        # A fibril can only fit at all if it fits along the cylinder's longest
        # direction - beyond that, whether a given orientation fits is decided
        # per-orientation inside the loop.
        fib_half_extent = np.max(np.linalg.norm(fib_template, axis=1))
        if fib_half_extent > max(cyl_radius, cyl_half_height):
            raise ValueError(f"cylinder too small to hold a fibril with {n_hets} Het-s beads")

        # Orientation is drawn FIRST, then the centre is bounded by that
        # orientation's ACTUAL extents. Bounding by fib_half_extent instead (a
        # sphere of ~half the fibril length, ~263nm here) applies the
        # worst-case orientation's margin in every direction at once, which
        # confined fibril centres to r <= 210nm of a 473nm cylinder - only
        # ~9% of its volume - while streptavidin (margin 3.78nm) filled the
        # whole thing. That made the system radially inhomogeneous: measured
        # biotin-end density fell ~12x from the axis to the wall while
        # streptavidin stayed flat, so the nominal 0.5uM state point did not
        # describe the material anywhere in the cylinder. A fibril pointing
        # along z only needs its length as clearance in z, not radially, and
        # bounding it that way recovers ~2.5x the accessible volume.
        #
        # The triangle inequality is what makes the per-axis bounds safe:
        # |bead_xy| <= |centre_xy| + |rotated_bead_xy| <= r_bound + radial
        # extent = cyl_radius, and likewise in z, so no bead can leave the
        # cylinder however the fibril is turned.
        for _ in range(max_attempts):
            rotated = fib_template @ quat_to_rotmat(random_unit_quaternion(rng)).T
            radial_extent = np.max(np.linalg.norm(rotated[:, :2], axis=1))
            axial_extent = np.max(np.abs(rotated[:, 2]))
            r_bound = cyl_radius - radial_extent
            h_bound = cyl_half_height - axial_extent
            if r_bound <= 0 or h_bound <= 0:
                continue  # this orientation cannot fit anywhere - redraw it
            # Accept the orientation in proportion to the volume it can
            # actually reach (V ~ r_bound^2 * h_bound). Without this,
            # drawing orientation and position independently would
            # over-represent the cramped orientations: an axial fibril has
            # ~2.2x the accessible volume of an in-plane one here, so equal
            # orientation sampling would bias the initial structure. That
            # matters more than usual in this system because fibrils barely
            # diffuse over the run (~4nm RMS in 3us), so the initial
            # orientation distribution is essentially the one being measured.
            if rng.random() > (r_bound ** 2 * h_bound) / (cyl_radius ** 2 * cyl_half_height):
                continue
            center = _sample_cylinder_center(rng, r_bound, h_bound)
            world = rotated + center
            p0, p1 = world[0], world[-1]
            if any(segment_segment_dist(p0, p1, q0, q1) < 2 * FIBRIL_EXCL_RADIUS
                   for q0, q1 in fibril_segments):
                continue
            break
        else:
            raise RuntimeError(
                f"could not place a fibril in {max_attempts} attempts - box too crowded"
            )
        fibril_segments.append((p0, p1))
        # Bonds only connect consecutive beads within THIS fibril's own
        # backbone - offset by where its beads start in the eventual global
        # particle array (everything already added from earlier fibrils).
        start = len(fibril_typeids)
        for i in range(len(fib_typeid) - 1):
            bond_type = BOND_TYPE_BY_PARTICLE_TYPEID_PAIR[_bond_key(fib_typeid[i], fib_typeid[i + 1])]
            fibril_bonds.append((start + i, start + i + 1, bond_type))
        for i in range(len(fib_typeid) - 2):
            angle_type = ANGLE_TYPE_BY_PARTICLE_TYPEID_WINDOW[
                _angle_key(fib_typeid[i], fib_typeid[i + 1], fib_typeid[i + 2])
            ]
            fibril_angles.append((start + i, start + i + 1, start + i + 2, angle_type))
        for i in range(len(fib_typeid) - 3):
            a, b, c, d = fib_typeid[i], fib_typeid[i + 1], fib_typeid[i + 2], fib_typeid[i + 3]
            if _is_degenerate_dihedral_window(a, b, c, d):
                continue
            dihedral_type = DIHEDRAL_TYPE_BY_PARTICLE_TYPEID_WINDOW[_dihedral_key(a, b, c, d)]
            fibril_dihedrals.append((start + i, start + i + 1, start + i + 2, start + i + 3, dihedral_type))

        fibril_positions.append(world)
        fibril_typeids.extend(fib_typeid)

    for _ in range(n_strep):
        for _ in range(max_attempts):
            center = _sample_cylinder_center(rng, strep_radius_bound, strep_height_bound)
            # core-to-core: twice the radius keeps both bodies' future
            # constituents clear of one another
            if any(np.linalg.norm(center - c) < 2 * STREP_EXCL_RADIUS
                   for c in strep_centers):
                continue
            if any(point_segment_dist(center, p0, p1) < STREP_EXCL_RADIUS + FIBRIL_EXCL_RADIUS
                   for p0, p1 in fibril_segments):
                continue
            break
        else:
            raise RuntimeError(
                f"could not place a streptavidin core in {max_attempts} attempts - box too crowded"
            )
        strep_centers.append(center)
        # Each core gets its own random orientation - it's the only thing
        # that will determine where its four not-yet-placed Strep_cons
        # constituents end up once hoomd.md.constrain.Rigid().create_bodies()
        # runs later (world_offset = R(orientation) @ local_offset), so
        # every core sharing one orientation would point all their binding
        # arms the same way, same artifact as unrotated parallel fibrils.
        strep_orientations.append(random_unit_quaternion(rng))

    positions = np.vstack([np.vstack(fibril_positions), np.array(strep_centers)])
    if np.any(np.abs(positions) > box_L / 2):
        raise RuntimeError("a bead landed outside the box - placement margins are wrong")
    radial = np.linalg.norm(positions[:, :2], axis=1)
    if np.any(radial > cyl_radius) or np.any(np.abs(positions[:, 2]) > cyl_half_height):
        raise RuntimeError("a bead landed outside the cylindrical wall - placement margins are wrong")

    # typeid[i] must describe positions[i] - built in the exact same order
    # (all fibril beads, fibril by fibril, then all streptavidin cores) as
    # positions was just stacked in above.
    typeid = np.array(fibril_typeids + [STREP_CENT] * n_strep, dtype=np.uint32)
    assert len(typeid) == len(positions)

    # orientation[i] follows the same order. Fibril beads get the identity
    # quaternion - their placement rotation already lives in positions (a
    # bonded chain's shape/orientation is fully expressed by where its beads
    # sit), and with isotropic moment_inertia and no anisotropic potential a
    # fibril bead's own orientation is physically inert. Strep_cent's is not
    # inert (see above), hence the real random draw there.
    n_fibril_beads = len(fibril_typeids)
    orientation = np.vstack([
        np.tile([1.0, 0.0, 0.0, 0.0], (n_fibril_beads, 1)),
        np.array(strep_orientations),
    ])
    assert len(orientation) == len(positions)

    # Streptavidin cores have no bonds at all, so bond_group/bond_typeid only
    # ever reference fibril-bead indices - already global since every
    # fibril's beads were stacked before any streptavidin core.
    bond_group = np.array([(i, j) for i, j, _ in fibril_bonds], dtype=np.uint32).reshape(-1, 2)
    bond_typeid = np.array([t for _, _, t in fibril_bonds], dtype=np.uint32)

    # Same reasoning for angles - streptavidin cores never appear here.
    angle_group = np.array(
        [(i, j, k) for i, j, k, _ in fibril_angles], dtype=np.uint32
    ).reshape(-1, 3)
    angle_typeid = np.array([t for *_, t in fibril_angles], dtype=np.uint32)

    # Same reasoning for dihedrals - streptavidin cores never appear here,
    # and degenerate (collinear Het-s) windows were already skipped above.
    dihedral_group = np.array(
        [(i, j, k, l) for i, j, k, l, _ in fibril_dihedrals], dtype=np.uint32
    ).reshape(-1, 4)
    dihedral_typeid = np.array([t for *_, t in fibril_dihedrals], dtype=np.uint32)

    return (
        positions, typeid, orientation,
        bond_group, bond_typeid,
        angle_group, angle_typeid,
        dihedral_group, dihedral_typeid,
    )


#gsd snapshot
init = gsd.hoomd.Frame()

# PLACEHOLDER fibril-length distribution - stand-in for the real per-fibril
# Het-s counts supplied by the rest of the project. Swap this out for that
# once it exists; randomise_positions only needs the resulting list/array.
_n_hets_per_fibril = np.maximum(
    1, np.round(np.random.default_rng(SIMULATION_SEED).normal(343, 20, size=N_FIBRIL))
).astype(int)

(
    position, typeid, orientation,
    bond_group, bond_typeid,
    angle_group, angle_typeid,
    dihedral_group, dihedral_typeid,
) = randomise_positions(_n_hets_per_fibril, N_STREP, L, CYL_RADIUS, CYL_HALF_HEIGHT,
                          seed=SIMULATION_SEED)
init.particles.N = position.shape[0]
init.particles.position = position
init.particles.types = TYPES
init.particles.typeid = typeid
init.particles.orientation = orientation

# mass[i] tracks typeid[i], same shape/order as typeid
init.particles.mass = np.array([MASS_BY_TYPEID[t] for t in typeid])

# moment_inertia[i] tracks typeid[i], same shape/order as typeid/mass
init.particles.moment_inertia = np.array([MOMENT_INERTIA_BY_TYPEID[t] for t in typeid])
# box: should bbe at least 4000 nm, because if we allow full gel
# expansion, it spans to around ~3730 nm max vert with 25 nodes

# bonds - only within each fibril's backbone; streptavidin cores have none
init.bonds.N = len(bond_group)
init.bonds.types = BOND_TYPES
init.bonds.typeid = bond_typeid
init.bonds.group = bond_group

# angles - only within each fibril's backbone; streptavidin cores have none
init.angles.N = len(angle_group)
init.angles.types = ANGLE_TYPES
init.angles.typeid = angle_typeid
init.angles.group = angle_group


# dihedrals - F-F-F-F and M-F-F-F (F=HETS) are neglected, because F-F-F is
# exactly collinear (see DIHEDRAL_TYPES section comment above); those windows
# are skipped entirely rather than merged into another type.
init.dihedrals.N = len(dihedral_group)
init.dihedrals.types = DIHEDRAL_TYPES
init.dihedrals.typeid = dihedral_typeid
init.dihedrals.group = dihedral_group

init.configuration.box = [L,L,L,0,0,0]
with gsd.hoomd.open(name="full_init.gsd", mode = "w") as f:
    f.append(init)


gpu = hoomd.device.GPU()
simulation = hoomd.Simulation(device=gpu, seed=SIMULATION_SEED)
simulation.create_state_from_gsd(filename="full_init.gsd")

# calcaulating orientation of constituents so their orientation matches the
# positional offset - same as sample_strep-biotin_sim_fixed.py
def quat_from_a_to_b(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)
    if dot < -1 + 1e-8:
        # a and b are anti-parallel - no unique axis, so pick any perpendicular one
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        return (0.0, *axis)

    w, xyz = 1 + dot, np.cross(a, b)
    q = np.array([w, *xyz])
    return tuple(q / np.linalg.norm(q))


director_rest_direction = (1, 0, 0)
cons_orientations = [quat_from_a_to_b(director_rest_direction, p) for p in STREP_CONS_POS]

# defining rigid geometries. STREP_CONS_POS is already in the principal-axis
# frame implied by STREP_CENT_MOMENT_INERTIA (see rigid_body_moment_inertia
# above) - each Strep_cent particle's own random orientation (set per-core
# during placement) is what create_bodies() applies on top of this shared
# body-local geometry to get each core's actual world-space constituents, so
# nothing further needs correcting per-core here.
rigid = hoomd.md.constrain.Rigid()
rigid.body["Strep_cent"] = {
    "constituent_types": ["Strep_cons", "Strep_cons", "Strep_cons", "Strep_cons"],
    "positions": list(map(tuple, STREP_CONS_POS)),
    "orientations": cons_orientations,
}

rigid.create_bodies(simulation.state)  # create the rigid body

integrator = hoomd.md.Integrator(dt=0.020, integrate_rotational_dof=True)
integrator.rigid = rigid

# ---------------------------------------------------------------------------
# non-bonded pairwise interactions. See all_pairwise_interactions and the
# categorisation it implements:
#   A. Strep_cons-Biotin: anisotropic directional binding (PatchyLJ) - the
#      one real high-affinity interaction in this system (CLAUDE.md).
#   B. everything else that can plausibly touch: isotropic WCA-truncated
#      steric exclusion (hoomd.md.pair.LJ, r_cut = sigma*2**(1/6)).
#   C. anything involving Strep_cent: non-interacting (epsilon=0) - it's
#      purely the rigid-body bookkeeping anchor, never a real surface.
# Real values are reused where they already exist in
# sample_strep-biotin_sim_fixed.py; everything else is a PLACEHOLDER - there
# is no derived data for it yet (see all_pairwise_interactions).
# ---------------------------------------------------------------------------
# Tree, not Cell: memory scales with particle count rather than box volume,
# which matters here since the box is much larger than the nonbonded cutoffs
# (~1-8nm) - Cell's grid-of-cells approach blows up in that regime and
# exhausts memory (confirmed empirically at the old L=4000nm).
tree = hoomd.md.nlist.Tree(buffer=0.4, exclusions=["bond", "angle", "dihedral", "body"])

# --- Category A: Strep_cons-Biotin binding (PatchyLJ) ---
# NOT a placeholder, and not slated for MS-IBI: the coarse-graining mapping
# makes this one exactly 1:1. One Strep_cons bead is one streptavidin
# MONOMER, which carries exactly one biotin site, and one Biotin bead is one
# biotin+amide - so a single bead pair is a single binding event and the
# measured per-site free energy transfers with no multiplicity correction.
#
# 80 kJ/mol is triple-anchored: all-atom streptavidin-biotin gives
# ~-18 kcal/mol = -75.3 kJ/mol; avidin -20.4 kcal/mol = -85.4 kJ/mol; and the
# experimental Kd = 4e-14 M gives dG = RT ln(Kd) = -76.4 kJ/mol at 298K. At
# 32.3 kT the bound-state lifetime is ~100s - utterly irreversible over a
# microsecond-scale run, which is what CLAUDE.md asks for ("model as
# strong/permanent bonds").
#
# This stays analytic even after MS-IBI lands: IBI targets a radial
# distribution function and so yields an ISOTROPIC potential, and HOOMD 7.1.2
# has no tabulated patchy potential (the aniso family - PatchyLJ, PatchyMie,
# PatchyGaussian, PatchyYukawa and the Expanded variants - is entirely
# analytic). An IBI curve for this pair therefore has to be FITTED to one of
# those forms. PatchyMie (tunable n/m exponents set the well width) and
# PatchyExpandedLJ (a delta shift sets the well position independently of
# sigma) are much better fit targets than plain PatchyLJ, whose minimum is
# locked to 2^(1/6)*sigma with a fixed shape.
PMF_DEPTH = 80  # kJ/mol, per binding site (see derivation above)
PMF_R_MIN = STREP_SUBUNIT_RADIUS + BIOTIN_RADIUS  # nm, bead contact distance
SIGMA_PATCH = PMF_R_MIN / 2 ** (1 / 6)

patches = hoomd.md.pair.aniso.PatchyLJ(nlist=tree, default_r_cut=1.0, mode="shift")
patches.params.default = dict(pair_params=dict(epsilon=0, sigma=1),
                               envelope_params=dict(alpha=math.pi / 4, omega=30))
# alpha=pi/12 keeps the binding directional. Note it does NOT by itself
# restrict a monomer to one biotin - a second biotin can sit 11 degrees
# off-axis (inside this cone) and still get 84% of the envelope. Single
# occupancy is enforced sterically instead, by BIOTIN_EXCL_RADIUS above.
patches.params[("Strep_cons", "Biotin")] = dict(
    pair_params=dict(epsilon=PMF_DEPTH, sigma=SIGMA_PATCH),
    envelope_params=dict(alpha=math.pi / 12, omega=30),
)
patches.r_cut[("Strep_cons", "Biotin")] = 2.5 * SIGMA_PATCH
patches.directors.default = []
patches.directors["Strep_cons"] = [(1, 0, 0)]
patches.directors["Biotin"] = [(1, 0, 0)]

# --- Categories B+C: everything else (isotropic WCA / non-interacting) ---
# This is the all_pairwise_interactions "volume exclusion" group: steric only,
# with no chemistry for MS-IBI to capture, so these values are FINAL rather
# than interim.
#
# For a purely repulsive WCA core, epsilon does nothing except set the cost of
# overlapping - so the only real requirement is epsilon >= kT, and there is no
# observable that would pin a per-pair value. Hence one uniform number rather
# than a size-scaled table, which would imply precision the model lacks.
#
# The previous epsilon=1 (inherited from sample_strep-biotin_sim_fixed.py,
# where it is likewise labelled a "soft steric placeholder") was too soft:
# with kT = 2.478 kJ/mol at 298K, reaching nominal contact r=sigma cost only
# 0.4 kT, so cores interpenetrated routinely. At 2.5 kJ/mol:
#     r = sigma        -> 1.0 kT
#     r = 0.9 sigma    -> 7.7 kT
#     r = 0.8 sigma    -> 44 kT
# Checked not to constrain the timestep: the fastest WCA mode (PEG-PEG) has a
# 743fs period, 3.3x slower than the PEG-PEG bond's limiting 226fs.
EPSILON_EXCL = 2.5  # kJ/mol, = kT at 298K (also brackets Martini's weakest
                     # interaction level, 2.0 kJ/mol)

lj = hoomd.md.pair.LJ(nlist=tree, default_r_cut=0.0, mode="shift")
lj.params.default = dict(epsilon=0, sigma=1)  # Category C - Strep_cent and
                                               # any other undeclared pair

def _set_wca(type_a, type_b, typeid_a, typeid_b, epsilon=EPSILON_EXCL):
    sigma = wca_sigma(typeid_a, typeid_b)
    lj.params[(type_a, type_b)] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[(type_a, type_b)] = sigma * 2 ** (1 / 6)

# Method (WCA truncation, sigma from bead radii) follows
# sample_strep-biotin_sim_fixed.py; the epsilon there is a placeholder too,
# not measured data, so it is not "reused" as a real value.
_set_wca("Biotin", "Biotin", BIOTIN, BIOTIN)  # uses BIOTIN_EXCL_RADIUS - see
                                               # the note there on enforcing
                                               # one biotin per strep monomer
_set_wca("Strep_cons", "Strep_cons", STREP_CONS, STREP_CONS)
_set_wca("Strep_cons", "Biotin", STREP_CONS, BIOTIN)  # baseline WCA,
                                                        # coexists with the
                                                        # PatchyLJ envelope
                                                        # above on this pair

# Remaining steric pairs. sigma is geometry-exact (bead radii); epsilon is the
# same uniform kT-scale core as above.
_set_wca("Biotin", "PEG", BIOTIN, PEG)
_set_wca("Biotin", "Malemide", BIOTIN, MALEMIDE)
_set_wca("Biotin", "HETS", BIOTIN, HETS)
_set_wca("PEG", "PEG", PEG, PEG)
_set_wca("PEG", "Malemide", PEG, MALEMIDE)
_set_wca("PEG", "HETS", PEG, HETS)
_set_wca("PEG", "Strep_cons", PEG, STREP_CONS)
_set_wca("Malemide", "Malemide", MALEMIDE, MALEMIDE)
_set_wca("Malemide", "HETS", MALEMIDE, HETS)
_set_wca("Malemide", "Strep_cons", MALEMIDE, STREP_CONS)
_set_wca("HETS", "HETS", HETS, HETS)
_set_wca("HETS", "Strep_cons", HETS, STREP_CONS)

# --- Weak nonspecific attraction on the streptavidin pairs ---
# INTERIM - replace with MS-IBI tables (hoomd.md.pair.Table). These are the
# all_pairwise_interactions "derive PMF" pairs; the values here are seeds that
# keep the model running, not predictions of what MS-IBI will converge to
# (an iterated IBI potential is generally SHALLOWER than its PMF seed, since
# iterating removes the indirect many-body correlations the PMF double-counts).
#
# Carried on a SECOND LJ object rather than by deepening the WCA above,
# because these pairs need a hard core AND a sub-kT well, and one LJ epsilon
# cannot set both independently. The WCA object above keeps supplying the
# core for every pair; this object adds only the attractive tail. HOOMD sums
# forces across integrator.forces - the same stacking the four
# dihedral.Periodic objects further below already rely on.
#
# The epsilons are small because of the COARSE-GRAINING MAPPING, which is easy
# to get wrong by an order of magnitude:
#  - Strep_cons is one streptavidin MONOMER, so a tetramer core is FOUR beads
#    and a core-core encounter sums over many bead pairs. Calibrating the
#    orientation-averaged association integral
#    K_a = 4*pi*N_A*int[<exp(-U/kT)>-1] r^2 dr over all 4x4 pairs, a per-bead
#    epsilon of 1.0-1.5 reproduces a core-core Kd of 10.5-1.8 mM, bracketing
#    ubiquitin's measured nonspecific self-association (Kd = 4.8 mM, the
#    standard "well-behaved soluble protein" reference). Treating it as a
#    single bead pair would have suggested ~3, which gives Kd = 0.3 mM -
#    about 16x too sticky, i.e. streptavidin would cluster.
#  - PEG is one EO unit, but the all-atom PEG-on-albumin reference reports
#    dG_ads = -1.5 to -2.8 kJ/mol per OLIGOMER (its chains are ~7 EO). Spread
#    over the units actually touching, and with 68 EO beads per fibril arm,
#    the per-bead value is ~0.5 kJ/mol. Anything larger glues PEG arms onto
#    streptavidin and blocks the biotin sites - backwards for a polymer whose
#    defining property is that it resists protein adsorption.
# All are free energies from solvated all-atom references, never gas-phase
# interaction energies: for PEG-albumin those differ by ~50x (-84 to -138
# kJ/mol vs -1.5 to -2.8), entirely desolvation, and this model is implicit
# solvent so the free energy is the correct input.
lj_attr = hoomd.md.pair.LJ(nlist=tree, default_r_cut=0.0, mode="shift")
lj_attr.params.default = dict(epsilon=0, sigma=1)  # inert on every other pair

def _set_attr(type_a, type_b, typeid_a, typeid_b, epsilon):
    sigma = wca_sigma(typeid_a, typeid_b)
    lj_attr.params[(type_a, type_b)] = dict(epsilon=epsilon, sigma=sigma)
    lj_attr.r_cut[(type_a, type_b)] = 2.5 * sigma  # full LJ, keeps the well

_set_attr("Strep_cons", "Strep_cons", STREP_CONS, STREP_CONS, 1.2)
_set_attr("Strep_cons", "HETS", STREP_CONS, HETS, 1.0)
_set_attr("Strep_cons", "PEG", STREP_CONS, PEG, 0.5)

# HETS-HETS and PEG-HETS deliberately get NO attractive term, and stay purely
# repulsive above. A fibril is ~344 HETS beads, so for any per-bead contact
# energy across the plausible range (1-40 kT), two fibrils lying side by side
# accumulate >300 kT and bundle irreversibly - which would out-compete
# streptavidin-mediated crosslinking and give fibril bundles instead of a
# node-crosslinked network. Revisit only if bundling is itself the object of
# study.

# ---------------------------------------------------------------------------
# Cylindrical wall (Eppendorf-tube interior) - confines the solution to the
# CYL_RADIUS/CYL_HALF_HEIGHT cylinder that the state point above was actually
# sized against (see the L/N_FIBRIL derivation above). This is what makes
# "no periodic boundaries" physically real rather than just a large box: the
# WCA wall keeps every particle away from the box faces, so the periodic
# images set up by L are never actually reached.
#
# hoomd.md.wall does not exist in HOOMD 7.1.2 - wall GEOMETRY lives in the
# top-level hoomd.wall module (Cylinder, Plane, Sphere), confirmed via
# dir(hoomd.wall); only the wall FORCE/potential classes live under
# hoomd.md.external.wall. A cylinder wall alone spans the whole box along its
# axis (unbounded), so two Plane walls cap it at +-CYL_HALF_HEIGHT to form a
# closed capsule matching the cylinder this file's state point assumes.
cylindrical_wall = hoomd.wall.Cylinder(radius=CYL_RADIUS, axis=(0, 0, 1))
top_cap = hoomd.wall.Plane(origin=(0, 0, CYL_HALF_HEIGHT), normal=(0, 0, -1))
bottom_cap = hoomd.wall.Plane(origin=(0, 0, -CYL_HALF_HEIGHT), normal=(0, 0, 1))

# WCA (purely repulsive, steric confinement only - no physisorption modelled)
# on every type that actually occupies space near the wall. sigma is set the
# same way as every other WCA sigma in this file - (radius_a+radius_b)/2^(1/6)
# - with the wall itself standing in for a particle of zero radius, so
# r_cut = sigma*2**(1/6) comes out to exactly the bead's own radius: the wall
# starts repelling right where the bead's surface would touch it.
# Strep_cent is excluded, consistent with it being non-interacting for every
# other pair in this file (it is the rigid-body bookkeeping anchor, never a
# real surface).
wallLJ = hoomd.md.external.wall.LJ(walls=[cylindrical_wall, top_cap, bottom_cap])
wallLJ.params.default = dict(epsilon=0, sigma=1, r_cut=0)

def _set_wall_wca(type_name, typeid_, epsilon=EPSILON_EXCL):
    sigma = RADIUS_BY_TYPEID[typeid_] / 2 ** (1 / 6)
    wallLJ.params[type_name] = dict(epsilon=epsilon, sigma=sigma, r_cut=sigma * 2 ** (1 / 6))

_set_wall_wca("Biotin", BIOTIN)
_set_wall_wca("PEG", PEG)
_set_wall_wca("Malemide", MALEMIDE)
_set_wall_wca("HETS", HETS)
_set_wall_wca("Strep_cons", STREP_CONS)


# bond potentials
har_bond = hoomd.md.bond.Harmonic()
har_bond.params["PEG-PEG"] = dict(k=17000, r0=0.33)
har_bond.params["HETS-HETS"] = dict(k=26.002*602.164, r0=1.4) # conversion from N/m to KJ/mol/nm^2

# INTERIM - replace with MS-IBI tables (hoomd.md.bond.Table); these 3
# junction bonds are exactly what all_pairwise_interactions slates for IBI
# ("PEG-malemide-prion: IBI for bond length, angle, and dihedral";
# "PEG-Biotin: IBI"). k is bracketed by the file's own two literature bonds
# (HETS-HETS 15657, PEG-PEG 17000) rather than invented, and is low-risk
# either way: bond stiffness sets the integration timestep, not the chain's
# conformational behaviour, which the angles and dihedrals control.
# r0 is exact - the bead-radius-sum (tangent-sphere) distance
# fibril_relative_pos already places these beads at, the same convention as
# RADIUS_BY_TYPEID/wca_sigma above, and independently validated by PEG-PEG
# where 2*0.17 = 0.34nm against the literature 0.33nm.
#
# Note for later: published Martini/PEO models Boltzmann-inverted from
# all-atom use b0=0.322nm with k=7000 (Rossi) where this file uses the
# Lee-type 0.33/17000. Both are literature; moving PEG-PEG to 7000 would
# raise the timestep ceiling by ~1.6x, since that bond is the limiting mode.
# r0 here is the researched chemistry-based bond length for each linkage
# (not the tangent-sphere radius-sum convention used elsewhere in this file -
# these are close to, but not identical to, BIOTIN_RADIUS+PEG_RADIUS etc).
# k=25000 for all three: stiffer than either literature bond (PEG-PEG 17000,
# HETS-HETS 15657), i.e. fractional bond fluctuation of only 1.0-2.0% of r0
# versus PEG-PEG's 3.7%. This does not threaten the timestep - PEG-PEG stays
# the limiting mode at 226fs (Biotin-PEG/PEG-Malemide/Malemide-HETS periods
# are 246/230/468fs at k=25000, checked against their reduced masses).
#
# INTERIM - seed for MSIBI refinement, not a fitted number; MSIBI measures
# these bond-length distributions directly (Boltzmann inversion of that
# distribution is exactly how a bonded term is derived), and solvent barely
# enters this estimate either way since bonded distributions are
# intramolecular and only weakly perturbed by hydration.
har_bond.params["Biotin-PEG"] = dict(k=25000, r0=0.55)
har_bond.params["PEG-Malemide"] = dict(k=25000, r0=0.50)
har_bond.params["Malemide-HETS"] = dict(k=25000, r0=1.02)

# angular potnentials. Two different angle-force classes are used here
# (CosineSquared for most types, Harmonic for the two exactly-180-degree
# ones - see the note below on why), so each needs a k=0 default covering
# the OTHER one's types: HOOMD requires every force object attached to the
# integrator to have params for every declared angle type, not just the
# ones it's actually meant to act on (same requirement already confirmed
# for dihedrals earlier).
cossqr_angle = hoomd.md.angle.CosineSquared()
cossqr_angle.params.default = dict(k=0, t0=0)
# Literature value, kept as-is. Flagged for later: k=85 at t0=130 implies a
# chain persistence length of ~0.73nm, against ~0.37nm from all-atom CHARMM
# C35r PEG (and 3.7-3.8 A experimentally) - about 2x stiff. Definitions of
# l_p differ between papers so this is a soft comparison, but it matters here
# because every junction angle below is anchored to this angle's effective
# stiffness.
cossqr_angle.params["PEG-PEG-PEG"] = dict(k=85, t0=np.radians(130))

J_per_rad2 = 6.08 * (10 ** (-17))
avogadros = 6.02214076 * (10 ** (23))  # precise value; previous 6.02e23 gave
                                        # 36601.6, this gives 36614.6
har_angle = hoomd.md.angle.Harmonic()
har_angle.params.default = dict(k=0, t0=0)
har_angle.params["HETS-HETS-HETS"] = dict(k=(J_per_rad2 * avogadros)/1000, t0=math.pi)

# INTERIM - replace with MS-IBI tables (hoomd.md.angle.Table); seeds for
# MSIBI refinement, not fitted numbers.
#
# Biotin-PEG-PEG and PEG-PEG-Malemide simply INHERIT PEG-PEG-PEG's literature
# parameters, rather than getting separate estimated values: an angle at a
# PEG bead is set by PEG's own chemistry and barely cares what sits two beads
# further away, so there is one measured angle in this chain, not three. This
# also retires the previous Biotin-PEG-PEG special case (Harmonic at t0=180,
# justified by "180 is degenerate for CosineSquared") - the geometry actually
# builds that junction at 130 degrees, so that justification did not hold in
# the first place.
cossqr_angle.params["Biotin-PEG-PEG"] = dict(k=85, t0=np.radians(130))
cossqr_angle.params["PEG-PEG-Malemide"] = dict(k=85, t0=np.radians(130))

# PEG-Malemide-HETS: t0=120 degrees keeps CosineSquared well away from its
# degenerate pole (sin^2(120)=0.75, no k compensation needed, unlike the
# near-180-degree junctions this replaces).
cossqr_angle.params["PEG-Malemide-HETS"] = dict(k=50, t0=np.radians(120))

# Malemide-HETS-HETS: Malemide reacts with Cystine mutation on beta solenoids 
# only on the ends of prion chains, so bond angle is set to 165 degrees, modelled
# as a harmmonic angle force
har_angle.params["Malemide-HETS-HETS"] = dict(k=150, t0=np.radians(165))

# Biotin-PEG-PEG-PEG and PEG-PEG-PEG-Malemide INHERIT the literature
# PEG-PEG-PEG-PEG 4-term set on every dihedralP1-P4 object, rather than
# getting a separate estimated value: torsion is about rotation around the
# CENTRAL bond (the 2nd-3rd bead), which for both of these is a PEG-PEG bond
# - identical to PEG-PEG-PEG-PEG - and only a terminal substituent differs.
dihedralP1 = hoomd.md.dihedral.Periodic()
dihedralP1.params["PEG-PEG-PEG-PEG"] = dict(k=1.96, d=1, n=1, phi0=np.radians(180))
dihedralP1.params["Biotin-PEG-PEG-PEG"] = dict(k=1.96, d=1, n=1, phi0=np.radians(180))
dihedralP1.params["PEG-PEG-PEG-Malemide"] = dict(k=1.96, d=1, n=1, phi0=np.radians(180))
dihedralP1.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP2 = hoomd.md.dihedral.Periodic()
dihedralP2.params["PEG-PEG-PEG-PEG"] = dict(k=0.18, d=1, n=2, phi0=np.radians(0))
dihedralP2.params["Biotin-PEG-PEG-PEG"] = dict(k=0.18, d=1, n=2, phi0=np.radians(0))
dihedralP2.params["PEG-PEG-PEG-Malemide"] = dict(k=0.18, d=1, n=2, phi0=np.radians(0))
dihedralP2.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP3 = hoomd.md.dihedral.Periodic()
dihedralP3.params["PEG-PEG-PEG-PEG"] = dict(k=0.33, d=1, n=3, phi0=np.radians(0))
dihedralP3.params["Biotin-PEG-PEG-PEG"] = dict(k=0.33, d=1, n=3, phi0=np.radians(0))
dihedralP3.params["PEG-PEG-PEG-Malemide"] = dict(k=0.33, d=1, n=3, phi0=np.radians(0))
dihedralP3.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP4 = hoomd.md.dihedral.Periodic()
dihedralP4.params["PEG-PEG-PEG-PEG"] = dict(k=0.12, d=1, n=4, phi0=np.radians(0))
dihedralP4.params["Biotin-PEG-PEG-PEG"] = dict(k=0.12, d=1, n=4, phi0=np.radians(0))
dihedralP4.params["PEG-PEG-PEG-Malemide"] = dict(k=0.12, d=1, n=4, phi0=np.radians(0))
dihedralP4.params.default = dict(k=0, d=0, n=0, phi0=0)

# INTERIM - replace with MS-IBI tables (hoomd.md.dihedral.Table). Seeds for
# MSIBI refinement, not fitted numbers. n=3 for both, the natural periodicity
# for rotation about a bond between sp3-like centres (vs the arbitrary n=1
# used previously). Carried on dihedralP3 alongside the real PEG n=3 term,
# rather than a new object, since HOOMD sums contributions across every
# force in integrator.forces regardless of which object they're declared on.
#
# Malemide-HETS-HETS-HETS and HETS-HETS-HETS-HETS are still excluded - 3
# consecutive HETS beads are exactly collinear (see DIHEDRAL_TYPES section
# above), a geometric degeneracy no parameter choice fixes.
dihedralP3.params["PEG-PEG-Malemide-HETS"] = dict(k=0.4, d=1, n=3, phi0=np.radians(0))
dihedralP3.params["PEG-Malemide-HETS-HETS"] = dict(k=0.3, d=1, n=3, phi0=np.radians(0))

integrator.forces.append(patches)
integrator.forces.append(lj)       # repulsive cores, every pair
integrator.forces.append(lj_attr)  # weak attractive tails, strep pairs only
integrator.forces.append(wallLJ)   # cylindrical + cap confinement
integrator.forces.append(har_bond)
integrator.forces.append(cossqr_angle)
integrator.forces.append(har_angle)
integrator.forces.append(dihedralP1)
integrator.forces.append(dihedralP2)
integrator.forces.append(dihedralP3)
integrator.forces.append(dihedralP4)

# ---------------------------------------------------------------------------
# Integration method: Langevin dynamics, implicit solvent (CLAUDE.md) - drag
# and thermal noise stand in for the solvent instead of explicit solvent
# particles. filter=Rigid(("center","free")) integrates every free particle
# (fibril beads) plus each rigid body's own central particle (Strep_cent) -
# not the constituents, which are slaved to their body's motion by the
# Rigid constraint and are never separately integrated.
# ---------------------------------------------------------------------------
# Same unit convention used throughout this file (length=nm, mass=g/mol,
# energy=kJ/mol - see har_bond/har_angle above), so KB comes out in
# kJ/(mol*K) and time in ps for free, matching
# sample_strep-biotin_sim_fixed.py's convention.
KB = 0.0083144621  # kJ/(mol*K)
TEMPERATURE_K = 298.0  # room temperature
kT = KB * TEMPERATURE_K

rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
langevin = hoomd.md.methods.Langevin(filter=rigid_centers_and_free, kT=kT)

# Per-type friction from Stokes-Einstein drag in an implicit solvent, rather
# than HOOMD's default gamma=1 for every type regardless of size (same fix
# sample_strep-biotin_sim_fixed.py makes, for the same reason: a uniform
# gamma would make every bead relax as fast as the smallest one, independent
# of its real size). gamma_trans = 6*pi*eta*r, gamma_rot = 8*pi*eta*r^3
# (SI Stokes drag), converted Pa*s -> g/mol/(nm*ps) the same way.
AVOGADRO = 6.02214076e23  # 1/mol
PAS_TO_SIM_VISCOSITY = AVOGADRO * 1e3 * 1e-9 * 1e-12  # Pa*s -> g/mol/(nm*ps)
SOLVENT_VISCOSITY_PAS = 8.9e-4  # water at 25 C
eta = SOLVENT_VISCOSITY_PAS * PAS_TO_SIM_VISCOSITY

for _gamma_type, _gamma_typeid in [
    ("Biotin", BIOTIN), ("PEG", PEG), ("Malemide", MALEMIDE), ("HETS", HETS),
]:
    _r = RADIUS_BY_TYPEID[_gamma_typeid]
    langevin.gamma[_gamma_type] = 6 * math.pi * eta * _r
    langevin.gamma_r[_gamma_type] = (8 * math.pi * eta * _r ** 3,) * 3

# STREP_EXCL_RADIUS is exactly this body's hydrodynamic radius - subunit
# radius plus how far a constituent sits from the body center - already
# computed above for placement clearance; reused here rather than
# redefining the same quantity under a new name.
langevin.gamma["Strep_cent"] = 6 * math.pi * eta * STREP_EXCL_RADIUS
langevin.gamma_r["Strep_cent"] = (8 * math.pi * eta * STREP_EXCL_RADIUS ** 3,) * 3

integrator.methods.append(langevin)

simulation.operations.integrator = integrator

# ---------------------------------------------------------------------------
# Energy minimisation, BEFORE thermalising.
#
# fibril_relative_pos builds an idealised, perfectly planar geometry whose
# bonded coordinates do not sit at the force field's minima: junction bonds
# start 0.02-0.03nm off their r0 (2-4.5 kT each), the PEG angles are built at
# 160 degrees against t0=130, and every dihedral starts at exactly 180 degrees
# while the 4-term PEG series minimises near -42 degrees. Measured, that is
# ~195,000 kJ/mol of potential energy at t=0, of which minimisation removes
# ~130,000 (converges to ~65,000 in ~1400 steps). Running the production
# integrator straight from that state would spend the opening of the
# trajectory violently relaxing built-in strain rather than sampling, and
# would put a large artificial energy pulse into the Langevin thermostat.
#
# FIRE needs its own integrator, and HOOMD allows a Rigid object (and a given
# force) to belong to only ONE integrator at a time - constructing FIRE while
# `integrator` still holds them raises
# "Rigid object can only belong to one integrator". So the forces and the
# rigid constraint are handed over to FIRE and handed back afterwards.
#
# ConstantVolume with no thermostat is the right method here: FIRE needs plain
# NVE dynamics to damp against, and using Langevin would inject exactly the
# random forces minimisation is trying to remove. dt is 4x smaller than the
# production dt for stability while forces are still large.
_production_forces = list(integrator.forces)
integrator.forces.clear()
integrator.rigid = None

# energy_tol=1e-7 (an earlier value here) failed to converge in 20000 steps
# on a real GPU run, while converging fine in ~1400 steps on this file's own
# CPU verification runs. Root cause: this system's converged PE is
# ~65,000 kJ/mol, and consumer GPUs (RTX 5090 included) run single/mixed
# precision by default (weak FP64 throughput), whose quantization step AT
# that PE magnitude is ~0.008 kJ/mol - about 77,000x larger than 1e-7.
# Delta-PE between steps can never quantize below that floor on such a
# build, so the criterion was likely unreachable there in floating point,
# not just slow - consistent with converging on (presumably double-
# precision) CPU but not on GPU.
#
# force_tol/angmom_tol were NOT the problem - confirmed directly: with them
# left at their original values, only energy_tol loosened, minimisation
# still converges at the same ~1400 steps to the same ~65,000 kJ/mol PE as
# before. A first attempt that loosened all three together (to 1e-1/1e-1/
# 1e-3) "converged" at step 1400 too, but to PE=183,900 - a FALSE early
# stop, not real relaxation (measured separately: |delta PE|/N fluctuates
# noisily between ~4e-3 and ~2e-1 even mid-descent, well before any true
# minimum, so a loose energy_tol can be satisfied by a transient plateau
# rather than genuine convergence). energy_tol=1e-4 is the value that
# empirically avoids both failure modes: reachable well above the float32
# floor above, and tight enough to still land at the real minimum.
fire = hoomd.md.minimize.FIRE(
    dt=0.005,
    force_tol=1e-2,
    angmom_tol=1e-2,
    energy_tol=1e-4,
    integrate_rotational_dof=True,
    forces=_production_forces,
    methods=[hoomd.md.methods.ConstantVolume(filter=rigid_centers_and_free)],
    rigid=rigid,
)
simulation.operations.integrator = fire

_MINIMISE_CHUNK = 200
_MINIMISE_MAX_STEPS = 20000  # ~14x the ~1400 steps this needed when measured
print("Energy Minimisation Start")
for _ in range(_MINIMISE_MAX_STEPS // _MINIMISE_CHUNK):
    if fire.converged:
        print("Energy Minimisation completed sucessfully")
        break
    simulation.run(_MINIMISE_CHUNK)
if not fire.converged:
    raise RuntimeError(
        f"FIRE did not converge in {_MINIMISE_MAX_STEPS} steps - the initial "
        "geometry is further from the force field's minima than expected, "
        "check for a bonded parameter that disagrees with fibril_relative_pos"
    )

# hand the forces and the rigid constraint back to the production integrator
fire.forces.clear()
fire.rigid = None
integrator.rigid = rigid
for _f in _production_forces:
    integrator.forces.append(_f)
simulation.operations.integrator = integrator

trajectory_writer = hoomd.write.GSD(
    filename="full_Sim_traj.gsd",
    trigger=hoomd.trigger.Periodic(15000),
    mode="wb",
)

simulation.operations.writers.append(trajectory_writer)

# ---------------------------------------------------------------------------
# Estimated time remaining, printed to the terminal periodically during the
# production run. Standard HOOMD pattern: a small class carrying a
# hoomd.logging.log-decorated property, added to a Logger and read out by a
# Table writer - not something specific to Ubuntu, this is the same on any
# platform HOOMD runs on (Linux only per CLAUDE.md's own install notes).
#
# Note the Loggable metaclass, not requires_run=True: that flag is for
# loggables that are themselves attached HOOMD objects (forces, computes)
# and expect an internal _attached flag Status does not have - using it here
# raised AttributeError.
# ---------------------------------------------------------------------------
class Status(metaclass=hoomd.logging.Loggable):
    def __init__(self, simulation):
        self.simulation = simulation

    @hoomd.logging.log(category="string")
    def etr(self):
        tps = self.simulation.tps
        if tps == 0:
            return "n/a"
        remaining_steps = self.simulation.final_timestep - self.simulation.timestep
        return str(datetime.timedelta(seconds=remaining_steps / tps))

    @hoomd.logging.log(category="string")
    def date(self):
        # wall-clock date this line was printed, DD/MM/YY - not the same as
        # etr (time REMAINING); useful on a multi-day run to see at a glance
        # when a given line was logged without cross-referencing a separate
        # terminal timestamp.
        return datetime.datetime.now().strftime("%d/%m/%y")

status = Status(simulation)
status_logger = hoomd.logging.Logger(categories=["scalar", "string"])
status_logger.add(simulation, quantities=["timestep", "tps"])
status_logger.add(status, quantities=["etr", "date"])
# Every 100000 steps (not the trajectory writer's 15000) - frequent enough to
# see live progress without spamming the terminal over a multi-day run.
status_table = hoomd.write.Table(
    trigger=hoomd.trigger.Periodic(50000), logger=status_logger
)
simulation.operations.writers.append(status_table)

simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
simulation.run(150000000) # for 3 microseconds, dt of 2 femotseconds
simulation.operations.writers.remove(trajectory_writer)


