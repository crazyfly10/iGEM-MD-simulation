# CLAUDE.md

Context for Claude Code working in this repository.

## Project

Early/bare repo (currently just `README.md`) for an iGEM coarse-grained MD simulation, built on **HOOMD-blue v7.1.2**, MD module only (`hoomd.md`; ignore `hoomd.hpmc`, `hoomd.mpcd`).

**System modeled**: a node-based hydrogel.
- **Nodes** = streptavidin (tetravalent, 4 biotin-binding sites each).
- **Fibrils/crosslinkers** = coarse-grained prion protein chains, functionalized at both ends with **PEG-biotin-maleimide** (maleimide end conjugates to the prion chain; biotin end binds a streptavidin node).
- Network forms via streptavidin-biotin binding (very high affinity, effectively near-irreversible on simulation timescales — model as strong/permanent bonds unless explicitly testing unbinding).
- **Dynamics**: Langevin dynamics, **implicit solvent** (solvent effects folded into friction/random-force terms, no explicit solvent particles) → use `hoomd.md.methods.Langevin` (or `ConstantVolume`/`ConstantPressure` integrator wrapped with a Langevin thermostat, depending on API usage) rather than NVE/explicit-solvent methods.
- Coarse-grained: beads represent groups of atoms/residues, not individual atoms — expect reduced/LJ-style units, not atomistic units.

Update this file with real directory layout, entry points, and run/test commands as the codebase grows.

## HOOMD-blue MD workflow

1. `hoomd.device.CPU()`/`GPU()` → 2. `hoomd.Simulation(device=...)` → 3. `State` populated from `hoomd.Snapshot` or GSD (`sim.create_state_from_gsd/snapshot`) → 4. `hoomd.md.Integrator` on `sim.operations.integrator`, with **forces** (`hoomd.md.pair.*`, `bond.*`, `angle.*`, `dihedral.*`) and **methods** (`hoomd.md.methods.*`, e.g. `Langevin`, `ConstantVolume`) applied to a `hoomd.filter.*` group → 5. writers/computes/loggers added to `sim.operations` (`hoomd.write.GSD`, `hoomd.md.compute.ThermodynamicQuantities`) → 6. `sim.run(n_steps)`.

Docs: use version-pinned `https://hoomd-blue.readthedocs.io/en/v7.1.2/` — API changed significantly across major versions (v2→v3→v4+ restructured around `Simulation`/`Operations`).

## Units

No fixed unit system — derived from 3 base units: **energy, length, mass** (time ∝ √(mass/energy)·length; force = energy/length; pressure = energy/length³). Given coarse-grained model, likely reduced/LJ units (bead diameter = length unit, ε = energy unit, etc.) rather than "MD units" (kJ/mol, nm, ps) — confirm/state the convention explicitly wherever constants are defined; HOOMD-blue does not catch unit mismatches.

## Installation

```bash
mamba install hoomd=7.1.2                                # auto-detect GPU
mamba install "hoomd=7.1.2=*gpu*" "cuda-version=12.9"     # force GPU
mamba install "hoomd=7.1.2=*cpu*"                         # force CPU-only
```
Pre-built: Linux x86-64, macOS x86-64/ARM64. MPI/custom builds require compiling from source (CMake + compiler [+CUDA]); see [glotzerlab-software](https://glotzerlab-software.readthedocs.io).

## GSD trajectory files (`gsd.hoomd`)

Native trajectory format, read/written via the separate `gsd` package's `gsd.hoomd` submodule (`https://gsd.readthedocs.io/en/v5.0.1/python-module-gsd.hoomd.html`) — use for analysis scripts and hand-built initial configs.

- `gsd.hoomd.open(filename, mode)` → `HOOMDTrajectory` (context manager). Modes: `r`, `r+`, `w`, `x`, `a`.
- `HOOMDTrajectory`: sequence of `Frame`s — index/iterate; `append()`, `extend()`, `flush()`, `truncate()`, `close()`, `len()`.
- `Frame` fields: `configuration` (box, timestep, dimensionality); `particles` (`ParticleData`: `position` N×3, `velocity`, `orientation` N×4, `typeid`, `types`, `mass`, `charge`, `diameter`, ang. momentum, moment of inertia); `bonds`/`angles`/`dihedrals`/`impropers`/`pairs` (topology, dims 2/3/3/4/4); `constraints`; `log` (custom per-frame dict).
- Write: build `gsd.hoomd.Frame`, populate numpy arrays, `traj.append(frame)` — only changed fields are stored per frame (static topology need not be rewritten each frame).
- Read logged series: `gsd.hoomd.read_log()`, with optional name/glob filtering.
- Prefer reading GSD output directly for analysis over re-running simulations.

## Visualization (`fresnel`)

[Fresnel](https://fresnel.readthedocs.io/en/stable/) is a path-tracing renderer for publication-quality, GPU/CPU-accelerated rendering of particle simulations (soft matter-oriented, integrates naturally with HOOMD-blue/GSD data). Docs: `https://fresnel.readthedocs.io/en/stable/`.

- **Backends**: `fresnel.Device` selects GPU (NVIDIA OptiX) or CPU (Intel Embree).
- **Workflow**: build a `fresnel.Scene` from geometry primitives (spheres, cylinders, convex polyhedra, meshes, polygons, boxes) → apply `fresnel.material.Material` (roughness, specularity, metallic) → set camera + lighting (preset modes like "cloudy"/"lightbox") → render.
- **Rendering modes**: `scene.preview()` for fast/interactive draft renders, `scene.pathtrace()` for high-quality final images with global illumination.
- **Interactive Jupyter view**: `fresnel.interact.SceneView(scene)` opens a Qt widget with click-and-drag camera rotation (requires PySide2/Qt; enable via `%gui qt` in Jupyter first). Call `view.setScene(scene)` after mutating the scene to refresh it.
- Useful for eyeballing initial configurations (e.g. rigid-body geometry, patch/director orientation) and trajectory frames before committing to a full run, and for producing figures from GSD trajectories.

## Conventions

- No build system/tests yet — when adding first scripts, use a simple `scripts/` layout + `requirements.txt`/`environment.yml` pinning `hoomd=7.1.2`; document the chosen structure here once decided.
- Output trajectories via `hoomd.write.GSD` (native format, tool-compatible: `gsd`, `freud`, OVITO).
- Verify exact API signatures against v7.1.2 docs rather than assuming stability across versions.
