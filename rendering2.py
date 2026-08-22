# This is not intended as a full tutorial on fresnel - see the fresnel user
# documentation (https://fresnel.readthedocs.io/) if you would like to learn more.

import math
import os
import warnings

import fresnel
import numpy
import packaging.version
import PIL.Image

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)


def _check_version():
    if (
        "version" not in dir(fresnel)
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )


def _unwrap_positions(frame):
    # hoomd wraps particles.position back into the primary box every step;
    # particles.image counts how many times each particle has crossed a
    # periodic boundary, so adding it back reconstructs the continuous
    # position instead of a position that jumps across the box on wrap.
    # (assumes an orthorhombic, untilted box - box[3:6] would need to be
    # accounted for otherwise.)
    L = numpy.array(frame.configuration.box[:3])
    return frame.particles.position[:] + frame.particles.image[:] * L


def _build_scene(frame):
    L = frame.configuration.box[0]
    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(
        scene, N=len(frame.particles.position), radius=0.3
    )
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([252 / 255, 209 / 255, 1 / 255]), roughness=0.5
    )
    geometry.outline_width = 0.04
    fresnel.geometry.Box(scene, [L, L, L, 0, 0, 0], box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(
            direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3
        ),
    ]
    #scene.camera = fresnel.camera.Orthographic(
        #position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 0.8 + 1
    #)
    scene.camera = fresnel.camera.Orthographic.fit(scene, look_at=(0, 0, 0), up=(0, 1, 0), height=L * 0.8 + 1)
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    return scene, geometry


def _samples():
    return 100 if "CI" in os.environ else 2000


def render(snapshot):
    """Render a single frame and return it as a PIL Image."""
    _check_version()
    scene, geometry = _build_scene(snapshot)
    geometry.position[:] = _unwrap_positions(snapshot)
    tracer.sample(scene, samples=_samples())
    return PIL.Image.fromarray(tracer.output[:], mode="RGBA")


def render_movie(trajectory, filename="movie.gif", duration=100):
    """Render a gsd.hoomd trajectory to an animated GIF file on disk.

    Returns the filename. The scene/geometry/camera are built once and
    reused across frames (only sphere positions change) - this assumes the
    box doesn't change size between frames (true for constant-volume runs;
    rebuild per frame if you add NPT).
    """
    if len(trajectory) == 0:
        raise ValueError("Trajectory contains no frames.")

    _check_version()
    scene, geometry = _build_scene(trajectory[0])
    samples = _samples()

    rendered_frames = []
    for frame in trajectory:
        geometry.position[:] = _unwrap_positions(frame)
        tracer.sample(scene, samples=samples)
        rendered_frames.append(
            PIL.Image.fromarray(tracer.output[:], mode="RGBA").convert("RGB")
        )

    rendered_frames[0].save(
        filename,
        format="GIF",
        save_all=True,
        append_images=rendered_frames[1:],
        duration=duration,
        loop=0,
    )
    return filename
