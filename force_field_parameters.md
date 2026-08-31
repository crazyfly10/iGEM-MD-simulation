# Force-field parameters in `full_sim_sample.py` — values and justification

Reference for where every force parameter came from, what is settled versus
interim, and which reasoning must be preserved if a value is revisited.

`kT = 2.478 kJ/mol` at 298 K throughout. Units: length nm, mass g/mol, energy
kJ/mol, so time comes out in ps.

---

## 1. The derivation chain

1. **Reference: all-atom MD in *explicit* solvent** — hydration shells, the
   hydrophobic effect and desolvation penalties are physically present.
2. **Map to CG sites** using this project's mapping: one bead per
   **streptavidin monomer**; per **1.5 Het-s units**; per **EO unit** of PEG;
   per **biotin + amide**; per **maleimide + amide**. Bead centres are the
   centroids of exactly those groups.
3. **Multistate IBI (MS-IBI)** targets the RDFs between those sites across
   several state points: seed `U_0(r) = -kT ln g(r)`, iterate
   `U_{i+1} = U_i + kT ln(g_i / g_target)`.
4. **Output is an implicit-solvent potential.** With no solvent particles in
   the CG model, all solvent-mediated effects are absorbed into the potential.

### Two consequences that govern every number below

**Use free energies, never gas-phase interaction energies.** Because the CG
model has no explicit water, the pair potential has to already contain the
desolvation cost. The gap is large enough to change conclusions — for all-atom
PEG on human serum albumin:

| quantity | value |
|---|---|
| interaction energy (vdW + electrostatic) | -84 to -138 kJ/mol |
| adsorption **free** energy `dG_ads` | **-1.5 to -2.8 kJ/mol** |

~50x, entirely desolvation. Feeding in the first number would make PEG
catastrophically sticky.

**A converged IBI potential is not the PMF.** `g(r)` counts how often two beads
sit at separation `r`, so it contains *indirect* structure — A and B are close
partly because a third bead crowded them there. Used directly as a *pair*
potential, the PMF would make the CG run generate that crowding **again** from
its own many-body arrangement, double-counting it and over-structuring `g(r)`.
IBI's iterations walk the potential back until CG `g(r)` matches the all-atom
target. The converged result is the *direct* pair interaction that gives
correct structure once the simulation supplies many-body effects itself, and is
typically **shallower** than the PMF seed. So the interim values below are
reasonable MS-IBI seeds, not predictions of what MS-IBI will converge to.

---

## 2. State point

Set first and deliberately, because MS-IBI potentials are derived at state
points and the all-atom reference states should bracket the production run.

| quantity | value |
|---|---|
| `[fibril]` | **0.5 uM** (verified 0.499 from the built snapshot) |
| `N_FIBRIL` / `N_STREP` | 200 / 600 |
| cylinder (wall interior) | R = 473 nm, H = 946 nm |
| box `L` | 986 nm = `2 * (R + 20)` |
| particles | 99,659 |

Sizing is on the **solution volume** (the cylinder interior, where the material
actually is), not the cube: `V_cyl = pi R^2 H = N / (N_A * 0.5e-6)`.

**Why the 20 nm margin matters.** HOOMD boxes are *always* periodic — there is
no "PBC off" switch. "No periodic boundaries" is achieved by keeping every
particle at least one interaction cutoff from the box face, which the wall does.
The margin comfortably exceeds the largest cutoff in the system (7.6 nm, the
`Strep_cons`-`Strep_cons` attractive tail). This is what removes the fibril
self-image problem: fibrils are up to ~584 nm long, far more than `L/2`, so this
model would be unusable under genuine PBC.

**Why N=200 and not 100 or 300.** At 0.5 uM, N=100 gives L=693 nm — smaller than
a fibril. N=300 hits 0.5 uM at 145,585 particles, which runs 1.3-3.5 days and
breaches a 2-day budget in the pessimistic case. N=200 hits 0.5 uM at 99,659
particles and 0.7-1.8 days (see section 7). Placement was verified to converge
at all of these; volume fraction is only ~0.006%, so density is not a packing
obstacle.

**Resolved**: the cylindrical wall (`hoomd.wall.Cylinder` + two `hoomd.wall.Plane`
caps, WCA via `hoomd.md.external.wall.LJ`) is implemented, and
`randomise_positions` now samples the cylinder interior directly (uniform-disk
sampling, same `sqrt(uniform)` method `sample_strep-biotin_sim_fixed.py` uses),
not the box. Verified: 0/99,659 particles land outside the cylinder, wall energy
is exactly 0 at t=0, and 500 steps of real dynamics show no blowup.

---

## 3. Settled parameters (not slated for MS-IBI)

### Steric exclusion — `EPSILON_EXCL = 2.5 kJ/mol`

The `all_pairwise_interactions` "volume exclusion" group. Steric only, so there
is no chemistry for IBI to capture and these are **final**.

For a purely repulsive WCA core, `epsilon` does nothing except set the overlap
penalty, so the only real requirement is `epsilon >= kT`. The previous value of
1 kJ/mol (inherited from `sample_strep-biotin_sim_fixed.py`, where it is
likewise labelled a "soft steric placeholder", **not** measured data) was too
soft:

| separation | U at eps=1 | U at eps=2.5 |
|---|---|---|
| r = sigma (nominal contact) | **0.4 kT** | 1.0 kT |
| r = 0.9 sigma | 3.1 kT | 7.7 kT |
| r = 0.8 sigma | 17.7 kT | 44.3 kT |

At 0.4 kT, cores interpenetrate routinely. 2.5 kJ/mol equals kT at 298 K and
also brackets Martini's weakest interaction level (2.0).

**Deliberately uniform, not a per-pair table.** No observable pins a per-pair
`epsilon` for a repulsive-only core, so a size-scaled table would imply
precision the model does not have. A Derjaguin contact-area scaling was computed
as a cross-check (it spans 2.5 to 25.3 across pairs) and rejected for that
reason.

Checked against the timestep: the fastest WCA mode (PEG-PEG) has a 743 fs
period, 3.3x slower than the PEG-PEG bond's limiting 226 fs.

### Streptavidin-biotin binding — `PMF_DEPTH = 80 kJ/mol`

Stays analytic and is **not** handed to MS-IBI. The mapping makes this one
exactly 1:1 — one `Strep_cons` bead is one streptavidin monomer carrying exactly
one biotin site, and one `Biotin` bead is one biotin+amide — so the measured
per-site free energy transfers with **no multiplicity correction**.

Triple-anchored:

| source | value |
|---|---|
| all-atom streptavidin-biotin | -18 kcal/mol = **-75.3 kJ/mol** |
| all-atom avidin-biotin | -20.4 kcal/mol = -85.4 kJ/mol |
| experimental `Kd = 4e-14 M`, `dG = RT ln Kd` | **-76.4 kJ/mol** |

80 sits mid-range. At 32.3 kT the bound-state lifetime is ~100 s — irreversible
over a microsecond run, which is what CLAUDE.md asks for ("model as
strong/permanent bonds").

It stays analytic because **IBI cannot produce it**: IBI targets a radial
distribution function and so yields an *isotropic* potential, and HOOMD 7.1.2
has no tabulated patchy potential — the aniso family (`PatchyLJ`, `PatchyMie`,
`PatchyGaussian`, `PatchyYukawa`, `Expanded` variants) is entirely analytic. An
IBI curve for this pair must be **fitted** to one of those forms. `PatchyMie`
(tunable `n`/`m` set the well width) and `PatchyExpandedLJ` (a `delta` shift sets
the well position independently of sigma) are much better fit targets than plain
`PatchyLJ`, whose minimum is locked to `2^(1/6) sigma` with a fixed shape.

### One biotin per streptavidin monomer — `BIOTIN_EXCL_RADIUS = 0.55 nm`

`alpha = pi/12` was chosen to restrict a monomer to a single biotin. **It does
not achieve that on its own.** Biotin binds at 2.1 nm from the `Strep_cons`
centre, and two biotins need only clear each other sterically, so at the default
0.40 nm radius (contact 0.80 nm) a second biotin sits
`asin(0.80 / (2 * 2.1)) = 11 deg` off-axis — *inside* the 15 deg cone — where it
still receives **84%** of the envelope, and binds.

Narrowing `alpha` barely helps, and is expensive:

| alpha | f(2nd biotin), omega=30 | omega=300 | both-patch alignment |
|---|---|---|---|
| 15 deg | 0.838 | 0.991 | 0.029% |
| 8 deg | 0.762 | 0.075 | 0.0024% (**12x worse capture rate**) |

Enforced **sterically** instead, by raising the Biotin-Biotin exclusion radius
to 0.55 nm (contact 1.10 nm), which pushes the second biotin to 15.2 deg —
outside the cone:

| biotin exclusion radius | BB contact | min 2nd-biotin angle | outside 15 deg cone? |
|---|---|---|---|
| 0.40 (previous) | 0.80 nm | 11.0 deg | no |
| **0.55** | **1.10 nm** | **15.2 deg** | **yes** |

Costs nothing in capture rate, keeps `alpha = pi/12`, leaves the binding
geometry untouched (still r = 2.1 nm), and mirrors the real mechanism — a pocket
that physically accommodates one biotin. Applied as a Biotin-Biotin-specific
sigma, **not** by changing `BIOTIN_RADIUS` globally, which would perturb mass,
inertia, the binding distance and every other biotin pair.

---

## 4. Interim values (MS-IBI will replace these)

All marked `# INTERIM - replace with MS-IBI table` in the source.

### Nonbonded attraction — the streptavidin pairs

Carried on a **second `hoomd.md.pair.LJ` object** (`lj_attr`), not by deepening
the WCA, because these pairs need a **hard core AND a sub-kT well** and one LJ
`epsilon` cannot set both independently. The WCA object supplies the core for
every pair; `lj_attr` adds only the attractive tail (`r_cut = 2.5 sigma`). HOOMD
sums forces across `integrator.forces`, the same stacking the four
`dihedral.Periodic` objects rely on.

| pair | epsilon | basis |
|---|---|---|
| `Strep_cons`-`Strep_cons` | **1.2** | calibrated so the 4-bead core reproduces ubiquitin-like nonspecific self-association, `Kd = 4.8 mM` |
| `Strep_cons`-`HETS` | **1.0** | same protein-protein class; a core touches 2-4 HETS beads, keeping the total a few kT and reversible |
| `Strep_cons`-`PEG` | **0.5** | all-atom PEG-albumin `dG_ads`, reduced to per-EO-bead |

Verified as implemented: the summed WCA + attractive potential gives a
tetramer-tetramer **`Kd = 3.7 mM`** against the 4.8 mM target — streptavidin
stays soluble and freely reversible.

### `HETS`-`HETS` and `PEG`-`HETS` — deliberately no attraction

A fibril is ~344 HETS beads, so for **any** per-bead contact energy across the
plausible 1-40 kT range, two fibrils lying side by side accumulate >300 kT and
bundle irreversibly. That would out-compete streptavidin-mediated crosslinking
and produce fibril bundles instead of a node-crosslinked network. The argument
is deliberately a *robustness* one and does not depend on pinning the per-bead
value. Revisit only if bundling is itself the object of study.

### Bonded junctions

| parameter | value | note |
|---|---|---|
| `BOND_K_PLACEHOLDER` | 15000 kJ/mol/nm^2 | bracketed by the file's own literature bonds (HETS-HETS 15657, PEG-PEG 17000); low-risk since bond stiffness sets the timestep, not conformation |
| junction `r0` | 0.57 / 0.52 / 1.05 nm | exact — the tangent-sphere (radius-sum) distance `fibril_relative_pos` places beads at, validated by PEG-PEG where 2 x 0.17 = 0.34 vs literature 0.33 |
| `DIHEDRAL_K_PLACEHOLDER` | 0.5 kJ/mol | inside the range of the file's real PEG Fourier terms (0.12-1.96) |

### Angles — specified by *effective* stiffness

Effective stiffness is `k * sin^2(t0)` for CosineSquared and `k` for Harmonic, so
a single nominal `k` produces very different physics depending on where `t0`
sits. The previous uniform `k=85` gave a **9x spread** that was an artifact of
the functional form, not a modelling decision — and made junctions onto the
heavy maleimide/HETS beads *floppier* than the PEG chain they hang off, which is
backwards.

All junctions are now anchored to `k_eff = 49.9 kJ/mol/rad^2`, the effective
stiffness of `PEG-PEG-PEG` — the only angle here with a literature value:

| angle | t0 | k before | k_eff before | **k now** | k_eff now |
|---|---|---|---|---|---|
| PEG-PEG-PEG *(literature, unchanged)* | 130 | 85 | 49.9 | 85 | 49.9 |
| Biotin-PEG-PEG *(Harmonic)* | 180 | **20000** | **20000** | **49.9** | 49.9 |
| PEG-PEG-Malemide | 155 | 85 | 15.2 | 279.3 | 49.9 |
| PEG-Malemide-HETS | 165 | 85 | 5.7 | 744.6 | 49.9 |
| Malemide-HETS-HETS | 165 | 85 | 5.7 | 744.6 | 49.9 |

The large nominal values are `sin^2(t0)` compensation, not extra stiffness;
resulting mode periods are 2678-10992 fs, far from the limiting 226 fs.

**`Biotin-PEG-PEG` at k=20000 was outright wrong.** All-atom CHARMM C35r gives
PEG a persistence length of 3.7 A (experiment 3.7-3.8 A):

| k | RMS angular fluctuation | implied local `l_p` |
|---|---|---|
| **20000** | **0.64 deg** | **2664 nm** |
| 49.9 | 12.75 deg | 6.7 nm |
| all-atom PEG target | — | **0.37 nm** |

A flexible PEG-biotin linker modelled as a rod stiffer than the entire 530 nm
fibril — and silently the timestep-limiting mode (157 fs period, tighter than
the PEG-PEG bond's 226 fs).

`Harmonic` is retained for `Biotin-PEG-PEG` and `HETS-HETS-HETS` because
CosineSquared's local stiffness `k sin^2(t0)` is exactly zero at t0 = 180 deg —
a genuinely degenerate equilibrium, not merely a soft one.

---

## 5. Mapping audit — two errors this caught

Both came from ignoring what a bead actually *represents*. The same trap applies
to MS-IBI output, so the reasoning is worth keeping.

**`Strep_cons` is one monomer, so a tetramer is FOUR beads.** A core-core
encounter is not one bead pair. Calibrating over all 4x4 bead pairs with the
orientation-averaged association integral
`K_a = 4 pi N_A integral[<exp(-U/kT)> - 1] r^2 dr`:

| per-bead attractive eps | core-core Kd |
|---|---|
| 0.5 | no net association |
| **1.0-1.5** | **10.5 -> 1.8 mM** (brackets the 4.8 mM target) |
| 2.5 (first-pass value) | 0.3 mM — **~16x too sticky** |

Treating the association as a single isolated bead pair had suggested 3.07.

**`PEG` is one EO unit, but the all-atom source reports per *oligomer*.** Its
chains are ~7 EO units, so `dG_ads = -1.5 to -2.8 kJ/mol` is per molecule, not
per bead. With 68 EO beads per fibril arm the multiplicity compounds:

| EO beads in surface contact | per-bead eps giving the observed oligomer `dG_ads` |
|---|---|
| 3 | 0.70 kJ/mol |
| 5 | 0.42 kJ/mol |
| 8 | 0.26 kJ/mol |

So ~0.5 kJ/mol, not the 2.0 first proposed. Anything larger glues PEG arms onto
streptavidin and blocks the biotin sites — backwards for a polymer whose
defining property is resisting protein adsorption.

---

## 6. Flagged but unchanged (literature values)

- **`PEG-PEG-PEG`, k=85, t0=130** implies a chain persistence length of
  **0.73 nm** against the all-atom/experimental **0.37 nm** — about 2x stiff
  (0.37 would need k ~ 8.3). `l_p` conventions differ between papers so this is
  a soft comparison, but it matters because every junction angle is anchored to
  this angle's effective stiffness.
- **`PEG-PEG` bond, k=17000, r0=0.33** is the Lee-type parameterisation;
  Martini/PEO models Boltzmann-inverted from all-atom use **b0=0.322, k=7000**
  (Rossi). Both are literature. Moving to 7000 would raise the timestep ceiling
  by ~1.6x — a free speedup, since this bond is the limiting mode.

---

## 7. Verification results

Measured at the final configuration (99,659 particles), not predicted:

| check | result |
|---|---|
| `sim.run(0)` | succeeds, no `IncompleteSpecificationError` |
| state point | `[fibril] = 0.499 uM`; 20 nm box margin > 7.6 nm max cutoff |
| parameter categories | cores all 2.5; tails only on the 3 strep pairs; `Strep_cent` pairs 0 |
| biotin single occupancy | BB contact 1.100 nm -> 2nd biotin at 15.2 deg, **excluded** |
| core-core calibration | **Kd = 3.7 mM** vs 4.8 mM target |
| angle stiffness | all junctions at `k_eff = 49.9` |
| dt stability | **20 fs stable, 30 fs stable, 40 fs ejects a particle** |
| kinetic temperature | 2.453 vs target 2.478 (within 1%, no drift) |
| throughput | 9.1 steps/s at 99,659 particles on CPU |

**Note on a prediction that did not hold**: fixing the 20000 angle was expected
to relax the timestep ceiling. It did not — the limit is unchanged at 30 fs
stable / 40 fs failing, because the PEG-PEG bond (226 fs period) becomes the
binding constraint once the angle is fixed, and lands at a similar threshold.

**Runtime estimate.** CPU throughput scales as **0.63x** from 50k to 100k
particles, better than the 0.50x linear assumption. Extrapolating to an RTX 5090
for the 150M-step run:

| assumption | runtime |
|---|---|
| pessimistic | 1.83 days |
| mid | 1.10 days |
| optimistic | 0.69 days |

This is an extrapolation from CPU measurements, not a GPU benchmark. Run a timed
`sim.run(1000)` on the actual hardware before committing to the full run.
