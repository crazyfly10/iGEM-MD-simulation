import gsd.hoomd
import numpy as np

CRASH_PARTICLE_TAG = 51370  # from the error message

with gsd.hoomd.open("full_Sim_traj.gsd", mode="r") as traj:
    n_frames = len(traj)
    print(f"trajectory has {n_frames} frames")

    last = traj[-1]
    types = list(last.particles.types)
    tid = last.particles.typeid[CRASH_PARTICLE_TAG]
    pos = last.particles.position[CRASH_PARTICLE_TAG]
    print(f"\nparticle {CRASH_PARTICLE_TAG}: type = {types[tid]}")
    print(f"position in LAST WRITTEN frame: {pos}")

    # track this particle's position over the last several frames to see the
    # blow-up developing (or confirm it was already fine right up to the end)
    print(f"\nposition of particle {CRASH_PARTICLE_TAG} over the last 10 frames:")
    n_back = min(10, n_frames)
    for i in range(n_frames - n_back, n_frames):
        f = traj[i]
        p = f.particles.position[CRASH_PARTICLE_TAG]
        print(f"  frame {i:4d} (step {f.configuration.step:>12,}): "
              f"pos = ({p[0]:8.2f}, {p[1]:8.2f}, {p[2]:8.2f})")

    # nearest neighbours to this particle in the LAST frame, broken down by
    # type - identifies which interaction (WCA core / PatchyLJ / attractive
    # tail / wall) was actually in play just before the crash
    all_pos = last.particles.position
    all_tid = last.particles.typeid
    diff = all_pos - pos
    dist = np.linalg.norm(diff, axis=1)
    order = np.argsort(dist)
    print(f"\n10 nearest OTHER particles to {CRASH_PARTICLE_TAG} in the last frame:")
    count = 0
    for idx in order:
        if idx == CRASH_PARTICLE_TAG:
            continue
        print(f"  tag={idx:6d} type={types[all_tid[idx]]:12s} dist={dist[idx]:8.3f} nm")
        count += 1
        if count >= 10:
            break

    # also check: how close is this particle to the cylinder wall itself?
    r_xy = np.linalg.norm(pos[:2])
    print(f"\nradial distance from axis: {r_xy:.2f} nm")
    print(f"|z|: {abs(pos[2]):.2f} nm")
