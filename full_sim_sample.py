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
# 1.4 nm in diamter for het-s so 0.7 nm diameter

import math
import pandas as pd
import gsd.hoomd
import hoomd
import numpy as np

def fibril_relative_pos():
    Hets_radius = 0.7 #nm
    N_HETs = 343
    Het_s_pos = np.empty((0,3))
    # align along y axis
    for i in range(-(N_HETs//2), N_HETs//2 + 1):
        Het_s_pos = np.append(Het_s_pos, [[0,2*Hets_radius*i,0]], axis = 0)

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
    rep_unit = 68
    peg_peg_ang = math.radians(110)
    left_pegs = np.array([P1_M_H_chain[0].copy()])
    right_pegs = np.array([P1_M_H_chain[-1].copy()])
    y_peg_d = 2 * peg_radius * math.cos(math.pi/4 - peg_peg_ang/2)
    z_peg_d = 2 * peg_radius * math.sin(math.pi/4 - peg_peg_ang/2)
    for i in range(rep_unit):
        if i % 2 == 0:
            v = np.array([0,-y_peg_d,-z_peg_d])
        else:
            v = np.array([0,-y_peg_d,z_peg_d])
        current = left_pegs[i].copy() + v
        left_pegs = np.vstack([
            left_pegs,
            current
        ])
    for i in range(rep_unit):
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

    # per-bead type, built in the same order as the vstack above so it stays
    # in sync automatically if rep_unit/N_HETs change
    fib_types = (
        ["Biotin"]
        + ["PEG"] * len(left_pegs)
        + ["PEG"]                      # Mal_peg_l
        + ["Mal"]                      # first_mal
        + ["HETS"] * len(Het_s_pos)
        + ["Mal"]                      # last_mal
        + ["PEG"]                      # Mal_peg_r
        + ["PEG"] * len(right_pegs)
        + ["Biotin"]
    )
    # linear backbone: bond every consecutive pair of beads
    fib_bonds = [(i, i + 1) for i in range(full_fib.shape[0] - 1)]

    return full_fib.shape[0], full_fib, fib_types, fib_bonds

initial_cons_pos = np.array(
    [[-1.202,-1.202,-1.202],
     [1.202,1.202,-1.202 ],
     [-1.202,1.202,1.202],
     [1.202,-1.202,1.202]]
)

# streptavidin node template. Mirrors sample_strep-biotin_sim_fixed.py's
# rigid body: only the central particle goes into this snapshot, the four
# Strep_cons constituents get created later by
# hoomd.md.constrain.Rigid().create_bodies() once the state is loaded.
STREP_SUBUNIT_MASS = 14500  # g/mol, per subunit
STREP_SUBUNIT_RADIUS = 1.7  # nm, per subunit

def streptavidin_template():
    I_ref = np.identity(3) * (2 / 5 * STREP_SUBUNIT_MASS * STREP_SUBUNIT_RADIUS**2)
    I_general = np.zeros((3, 3))
    for r in initial_cons_pos:
        I_general += I_ref + STREP_SUBUNIT_MASS * (
            np.dot(r, r) * np.identity(3) - np.outer(r, r)
        )
    I_diagonal = np.real(np.linalg.eigvals(I_general))
    mass = 4 * STREP_SUBUNIT_MASS
    body_radius = STREP_SUBUNIT_RADIUS + np.linalg.norm(initial_cons_pos[0])
    return mass, I_diagonal, body_radius


# ---------------------------------------------------------------------------
# random rotation helpers
# ---------------------------------------------------------------------------
def random_unit_quaternion(rng):
    # Shoemake's method for a uniform-random rotation; returned in HOOMD's
    # scalar-first (w, x, y, z) convention.
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    y = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array([w, x, y, z])

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ])


# ---------------------------------------------------------------------------
# geometric overlap tests. Fibrils are approximated as capsules running
# between their first and last bead - the internal zigzag is only ~1 nm,
# negligible against the ~500 nm backbone, so this is accurate enough to
# avoid gross overlap at insertion time; any leftover fine-scale contact is
# left for the pair potentials to resolve once the run starts.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# random system assembly
# ---------------------------------------------------------------------------
N_STREP = 25    # streptavidin nodes
N_FIBRIL = 40   # crosslinking fibrils - placeholder count, tune to the
                # target node:crosslinker stoichiometry once the network
                # design is settled

L = 4000  # nm - box: should be at least 4000 nm, because if we allow full
          # gel expansion, it spans to around ~3730 nm max vert with 25 nodes

FIBRIL_CAPSULE_RADIUS = 1.0  # nm, placement-only clearance margin around the
                              # fibril backbone (largest real bead, biotin, is
                              # 0.4 nm - this is a generous safety margin)

# Placeholder bead masses (g/mol). BIOTIN reuses the real molar mass from
# sample_strep-biotin_sim_fixed.py, but note that file uses
# BIOTIN_RADIUS=0.25 nm for the free-biotin bead while biotin_radius above is
# 0.4 nm - reconcile these before merging the two pipelines, since "Biotin"
# needs one consistent size if it's meant to be the same species (and
# therefore bind the same Strep_cons patch) in both. Mal/PEG/HETS have no
# real molar-mass estimates yet.
BIOTIN_MASS = 244.31  # g/mol
FIBRIL_BEAD_MASS = {
    "Biotin": BIOTIN_MASS,
    "Mal": 125.0,    # PLACEHOLDER
    "PEG": 44.0,     # PLACEHOLDER, ~1 ethylene-glycol repeat unit
    "HETS": 1000.0,  # PLACEHOLDER, ~1.5 Het-s protein units per bead
}

TYPES = ["Strep_cent", "Strep_cons", "Biotin", "Mal", "PEG", "HETS"]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}


def build_random_system(n_strep, n_fibril, box_L, seed=0, max_attempts=5000):
    rng = np.random.default_rng(seed)

    fib_n, fib_template, fib_bead_types, fib_bonds_local = fibril_relative_pos()
    strep_mass, strep_I, strep_radius = streptavidin_template()

    # Max distance from a molecule's placement reference point to any of its
    # own beads (rotation-invariant) - centers must stay this far from the
    # box edge or beads could land outside the box, which HOOMD rejects, or
    # (if wrapped per-particle rather than per-molecule) tear a bonded chain
    # apart across the periodic boundary.
    fib_half_extent = np.max(np.linalg.norm(fib_template, axis=1))
    strep_half_extent = strep_radius
    fib_bound = box_L / 2 - fib_half_extent
    strep_bound = box_L / 2 - strep_half_extent
    if fib_bound <= 0 or strep_bound <= 0:
        raise ValueError("box_L too small to fit a fibril/streptavidin node without touching the boundary")

    strep_centers = []
    fibril_segments = []  # (p0, p1) world-space endpoints, per placed fibril

    positions, typeids, orientations = [], [], []
    masses, moment_inertias, bonds = [], [], []
    n_placed = 0

    for _ in range(n_strep):
        for _ in range(max_attempts):
            center = rng.uniform(-strep_bound, strep_bound, size=3)
            if any(np.linalg.norm(center - c) < 2 * strep_radius for c in strep_centers):
                continue
            if any(point_segment_dist(center, p0, p1) < strep_radius + FIBRIL_CAPSULE_RADIUS
                   for p0, p1 in fibril_segments):
                continue
            break
        else:
            raise RuntimeError(f"could not place streptavidin node after {max_attempts} attempts - box too crowded")

        strep_centers.append(center)
        positions.append(center)
        typeids.append(TYPE_INDEX["Strep_cent"])
        orientations.append(random_unit_quaternion(rng))
        masses.append(strep_mass)
        moment_inertias.append(strep_I)
        n_placed += 1

    for _ in range(n_fibril):
        for _ in range(max_attempts):
            center = rng.uniform(-fib_bound, fib_bound, size=3)
            quat = random_unit_quaternion(rng)
            world = fib_template @ quat_to_rotmat(quat).T + center
            p0, p1 = world[0], world[-1]
            if any(point_segment_dist(c, p0, p1) < strep_radius + FIBRIL_CAPSULE_RADIUS
                   for c in strep_centers):
                continue
            if any(segment_segment_dist(p0, p1, q0, q1) < 2 * FIBRIL_CAPSULE_RADIUS
                   for q0, q1 in fibril_segments):
                continue
            break
        else:
            raise RuntimeError(f"could not place fibril after {max_attempts} attempts - box too crowded")

        fibril_segments.append((p0, p1))
        start = n_placed
        positions.extend(world)
        typeids.extend(TYPE_INDEX[t] for t in fib_bead_types)
        orientations.extend([[1, 0, 0, 0]] * fib_n)
        masses.extend(FIBRIL_BEAD_MASS[t] for t in fib_bead_types)
        moment_inertias.extend([[0, 0, 0]] * fib_n)
        bonds.extend((start + a, start + b) for a, b in fib_bonds_local)
        n_placed += fib_n

    positions = np.array(positions)
    # Containment is already guaranteed by the margin-constrained sampling
    # above (fib_bound/strep_bound); wrapping post hoc would fold only the
    # occasional stray bead to the far side of the box and tear its bonds
    # apart, so this is a defensive check, not a fixup.
    if np.any(np.abs(positions) > box_L / 2):
        raise RuntimeError("a placed bead ended up outside the box - fib_half_extent/strep_half_extent margins are wrong")

    return {
        "N": n_placed,
        "position": positions,
        "typeid": typeids,
        "orientation": orientations,
        "mass": masses,
        "moment_inertia": moment_inertias,
        "bonds": np.array(bonds, dtype=np.uint32).reshape(-1, 2),
    }


system = build_random_system(N_STREP, N_FIBRIL, L)

#gsd snapshot
init = gsd.hoomd.Frame()
init.particles.N = system["N"]
init.particles.position = system["position"]
init.particles.orientation = system["orientation"]
init.particles.typeid = system["typeid"]
init.particles.types = TYPES
init.particles.mass = system["mass"]
init.particles.moment_inertia = system["moment_inertia"]
init.bonds.N = len(system["bonds"])
init.bonds.group = system["bonds"]
init.bonds.types = ["fibril_bond"]
init.bonds.typeid = [0] * len(system["bonds"])
init.configuration.box = [L, L, L, 0, 0, 0]

with gsd.hoomd.open(name="full_init.gsd", mode = "w") as f:
    f.append(init)





    
