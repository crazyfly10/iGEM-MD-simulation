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
import pandas as pd
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

    first_mal = np.array([Het_s_pos[0]]) + np.array([0,-y_mal,-z_mal])
    last_mal = np.array([Het_s_pos[-1]]) + np.array([0,y_mal,z_mal])
    Mal_w_pri = np.vstack([
        first_mal,
        Het_s_pos,
        last_mal,
    ])

    # add PEG-mal
    peg_radius = 0.17
    z_fpeg = (mal_radius + peg_radius) * math.sin(math.pi/4 - P_M_H_ang/2)
    y_fpeg = (mal_radius + peg_radius) * math.cos(math.pi/4 - P_M_H_ang/2)
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
    l_biotin = np.array(full_peg_M_H[0]) + np.array([0,-B_y_d, -B_z_d])
    r_biotin = np.array(full_peg_M_H[-1]) + np.array([0,B_y_d, -B_z_d])
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

L = 4000 #nm

# ---------------------------------------------------------------------------
# randomised placement of fibrils and streptavidin cores in the box
# ---------------------------------------------------------------------------
N_FIBRIL = 100
N_STREP = 300

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

def wca_sigma(typeid_a, typeid_b):
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


def randomise_positions(n_hets_per_fibril, n_strep, box_L, seed=0, max_attempts=10000):
    # n_hets_per_fibril: sequence of Het-s bead counts, one entry per
    # fibril to place (its length sets how many fibrils get placed). Real
    # fibrils aren't all the same length, so this comes from whatever
    # length distribution the rest of the project supplies rather than
    # being generated in here.
    rng = np.random.default_rng(seed)

    strep_bound = box_L / 2 - STREP_EXCL_RADIUS
    if strep_bound <= 0:
        raise ValueError("box_L too small to hold a streptavidin core")

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

        # Furthest any bead sits from this fibril's placement reference
        # point. The centre is held this far inside the wall so no bead
        # lands outside the box - a per-particle wrap would fold isolated
        # beads onto the opposite face and tear the chain apart, so it is
        # prevented rather than patched. Recomputed per fibril since length
        # (and therefore this margin) now varies fibril to fibril.
        fib_half_extent = np.max(np.linalg.norm(fib_template, axis=1))
        fib_bound = box_L / 2 - fib_half_extent
        if fib_bound <= 0:
            raise ValueError(f"box_L too small to hold a fibril with {n_hets} Het-s beads")

        for _ in range(max_attempts):
            center = rng.uniform(-fib_bound, fib_bound, size=3)
            world = fib_template @ quat_to_rotmat(random_unit_quaternion(rng)).T + center
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
            center = rng.uniform(-strep_bound, strep_bound, size=3)
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
    1, np.round(np.random.default_rng(0).normal(343, 20, size=N_FIBRIL))
).astype(int)

(
    position, typeid, orientation,
    bond_group, bond_typeid,
    angle_group, angle_typeid,
    dihedral_group, dihedral_typeid,
) = randomise_positions(_n_hets_per_fibril, N_STREP, L)
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
simulation = hoomd.Simulation(device=gpu, seed = 0)
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
# which matters here since the box (L=4000nm) is much larger than the
# nonbonded cutoffs (~1-5nm) - Cell's grid-of-cells approach blows up to
# ~490M cells in that regime and exhausts memory (confirmed empirically).
tree = hoomd.md.nlist.Tree(buffer=0.4, exclusions=["bond", "angle", "dihedral", "body"])

# --- Category A: Strep_cons-Biotin binding (PatchyLJ) ---
# Method reused from sample_strep-biotin_sim_fixed.py's PMF-derived approach
# (well depth, envelope shape - not that file's own BIOTIN_RADIUS, see the
# note on BIOTIN_RADIUS above), still provisional pending a real bead-level
# PMF calculation.
PMF_DEPTH = 80  # kJ/mol, placeholder well depth
PMF_R_MIN = STREP_SUBUNIT_RADIUS + BIOTIN_RADIUS  # nm, bead contact distance
SIGMA_PATCH = PMF_R_MIN / 2 ** (1 / 6)

patches = hoomd.md.pair.aniso.PatchyLJ(nlist=tree, default_r_cut=1.0, mode="shift")
patches.params.default = dict(pair_params=dict(epsilon=0, sigma=1),
                               envelope_params=dict(alpha=math.pi / 4, omega=30))
patches.params[("Strep_cons", "Biotin")] = dict(
    pair_params=dict(epsilon=PMF_DEPTH, sigma=SIGMA_PATCH),
    envelope_params=dict(alpha=math.pi / 12, omega=30),
)
patches.r_cut[("Strep_cons", "Biotin")] = 2.5 * SIGMA_PATCH
patches.directors.default = []
patches.directors["Strep_cons"] = [(1, 0, 0)]
patches.directors["Biotin"] = [(1, 0, 0)]

# --- Categories B+C: everything else (isotropic WCA / non-interacting) ---
EPSILON_EXCL = 1  # kJ/mol scale, generic steric strength placeholder -
                   # matches sample_strep-biotin_sim_fixed.py's convention

lj = hoomd.md.pair.LJ(nlist=tree, default_r_cut=0.0, mode="shift")
lj.params.default = dict(epsilon=0, sigma=1)  # Category C - Strep_cent and
                                               # any other undeclared pair

def _set_wca(type_a, type_b, typeid_a, typeid_b, epsilon=EPSILON_EXCL):
    sigma = wca_sigma(typeid_a, typeid_b)
    lj.params[(type_a, type_b)] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[(type_a, type_b)] = sigma * 2 ** (1 / 6)

# real values, reused from sample_strep-biotin_sim_fixed.py
_set_wca("Biotin", "Biotin", BIOTIN, BIOTIN)
_set_wca("Strep_cons", "Strep_cons", STREP_CONS, STREP_CONS)
_set_wca("Strep_cons", "Biotin", STREP_CONS, BIOTIN)  # baseline WCA,
                                                        # coexists with the
                                                        # PatchyLJ envelope
                                                        # above on this pair

# PLACEHOLDER - no derived data exists yet for any of these (see
# all_pairwise_interactions); sigma is geometry-correct (bead radii), only
# epsilon is a guess.
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


# bond potentials
har_bond = hoomd.md.bond.Harmonic()
har_bond.params["PEG-PEG"] = dict(k=17000, r0=0.33)
har_bond.params["HETS-HETS"] = dict(k=26.002*602.164, r0=1.4) # conversion from N/m to KJ/mol/nm^2

# PLACEHOLDER - no literature/IBI-derived stiffness exists yet for these 3
# junction bond types (unlike PEG-PEG/HETS-HETS above). k borrows the same
# order of magnitude as those two real values rather than an arbitrary
# scale; r0 is exact - the actual bead-radius-sum distance
# fibril_relative_pos already places these beads at, same convention as
# RADIUS_BY_TYPEID/wca_sigma above.
BOND_K_PLACEHOLDER = 15000  # kJ/mol/nm^2
har_bond.params["Biotin-PEG"] = dict(k=BOND_K_PLACEHOLDER, r0=BIOTIN_RADIUS + PEG_RADIUS)
har_bond.params["PEG-Malemide"] = dict(k=BOND_K_PLACEHOLDER, r0=PEG_RADIUS + MALEMIDE_RADIUS)
har_bond.params["Malemide-HETS"] = dict(k=BOND_K_PLACEHOLDER, r0=MALEMIDE_RADIUS + HETS_RADIUS)

# angular potnentials. Two different angle-force classes are used here
# (CosineSquared for most types, Harmonic for the two exactly-180-degree
# ones - see the note below on why), so each needs a k=0 default covering
# the OTHER one's types: HOOMD requires every force object attached to the
# integrator to have params for every declared angle type, not just the
# ones it's actually meant to act on (same requirement already confirmed
# for dihedrals earlier).
cossqr_angle = hoomd.md.angle.CosineSquared()
cossqr_angle.params.default = dict(k=0, t0=0)
cossqr_angle.params["PEG-PEG-PEG"] = dict(k=85, t0=np.radians(130))

J_per_rad2 = 6.08 * (10 ** (-17))
avogadros = 6.02 * (10 ** (23))
har_angle = hoomd.md.angle.Harmonic()
har_angle.params.default = dict(k=0, t0=0)
har_angle.params["HETS-HETS-HETS"] = dict(k=(J_per_rad2 * avogadros)/1000, t0=math.pi)

# PLACEHOLDER - no literature-derived stiffness/equilibrium exists yet for
# these 4 junction angle types (unlike PEG-PEG-PEG/HETS-HETS-HETS above). t0
# is the actual angle measured from fibril_relative_pos's own geometry
# (both ends' junctions, averaged for PEG-Malemide-HETS since the two ends
# don't land at quite the same angle: 150 vs 180 degrees).
#
# CosineSquared vs Harmonic: U_cossq(theta) = 1/2 k (cos(theta)-cos(t0))^2
# has local stiffness d^2U/dtheta^2 at theta=t0 equal to k*sin^2(t0), not
# just k - it vanishes as t0 -> 180 degrees (exactly zero AT 180, since
# sin(180)=0), a genuinely degenerate equilibrium, not just "softer".
# Biotin-PEG-PEG sits exactly at t0=180 degrees, so it keeps Harmonic, same
# reasoning as HETS-HETS-HETS above. The other three (155-165 degrees) are
# NOT at that pole - sin^2(t0) there is small but nonzero (~0.07-0.18), so
# CosineSquared is well-behaved (if intentionally softer than its nominal k
# suggests) and is the standard choice for coarse-grained bond angles
# (HOOMD's own docs: "CosineSquared is used in the gromos96 and MARTINI
# force fields") - matches PEG-PEG-PEG's own form/k order of magnitude
# above rather than mixing in a separately-tuned k for no better reason
# than it being placeholder either way.
ANGLE_K_PLACEHOLDER = 85  # kJ/mol, same order of magnitude as PEG-PEG-PEG's
                           # real CosineSquared value above
har_angle.params["Biotin-PEG-PEG"] = dict(k=20000, t0=math.radians(180))
cossqr_angle.params["PEG-PEG-Malemide"] = dict(k=ANGLE_K_PLACEHOLDER, t0=math.radians(155))
cossqr_angle.params["PEG-Malemide-HETS"] = dict(k=ANGLE_K_PLACEHOLDER, t0=math.radians(165))
cossqr_angle.params["Malemide-HETS-HETS"] = dict(k=ANGLE_K_PLACEHOLDER, t0=math.radians(165))

dihedralP1 = hoomd.md.dihedral.Periodic()
dihedralP1.params["PEG-PEG-PEG-PEG"] = dict(k=1.96, d=1, n=1, phi0=np.radians(180))
dihedralP1.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP2 = hoomd.md.dihedral.Periodic()
dihedralP2.params["PEG-PEG-PEG-PEG"] = dict(k=0.18, d=1, n=2, phi0=np.radians(0))
dihedralP2.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP3 = hoomd.md.dihedral.Periodic()
dihedralP3.params["PEG-PEG-PEG-PEG"] = dict(k=0.33, d=1, n=3, phi0=np.radians(0))
dihedralP3.params.default = dict(k=0, d=0, n=0, phi0=0)

dihedralP4 = hoomd.md.dihedral.Periodic()
dihedralP4.params["PEG-PEG-PEG-PEG"] = dict(k=0.12, d=1, n=4, phi0=np.radians(0))
dihedralP4.params.default = dict(k=0, d=0, n=0, phi0=0)

# PLACEHOLDER - no literature Fourier decomposition exists yet for these 4
# junction dihedral types (unlike PEG-PEG-PEG-PEG's real 4-term OPLS-style
# sum above, split across dihedralP1-P4). Rather than leaving them at the
# k=0 default (i.e. no torsional preference at all), a single n=1 term is
# added here on dihedralP4 as a minimal placeholder restraint - k borrows
# the same order of magnitude as the real PEG terms above (0.12-1.96); d=1,
# phi0=0 gives one cis-eclipsed minimum, an arbitrary but harmless choice
# pending real data. dihedralP1-P3 keep these types at their k=0 default,
# so the placeholder torsion is contributed once, not quadruple-counted.
DIHEDRAL_K_PLACEHOLDER = 0.5  # kJ/mol
dihedralP4.params["Biotin-PEG-PEG-PEG"] = dict(k=DIHEDRAL_K_PLACEHOLDER, d=1, n=1, phi0=np.radians(0))
dihedralP4.params["PEG-PEG-PEG-Malemide"] = dict(k=DIHEDRAL_K_PLACEHOLDER, d=1, n=1, phi0=np.radians(0))
dihedralP4.params["PEG-PEG-Malemide-HETS"] = dict(k=DIHEDRAL_K_PLACEHOLDER, d=1, n=1, phi0=np.radians(0))
dihedralP4.params["PEG-Malemide-HETS-HETS"] = dict(k=DIHEDRAL_K_PLACEHOLDER, d=1, n=1, phi0=np.radians(0))

integrator.forces.append(patches)
integrator.forces.append(lj)
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
# energy=kJ/mol - see BOND_K_PLACEHOLDER/har_angle above), so KB comes out in
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

trajectory_writer = hoomd.write.GSD(
    filename="full_Sim_traj.gsd",
    trigger=hoomd.trigger.Periodic(40000),
    mode="wb",
)

simulation.operations.writers.append(trajectory_writer)

simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
simulation.run(150000000) # for 3 microseconds, dt of 2 femotseconds
simulation.operations.writers.remove(trajectory_writer)


