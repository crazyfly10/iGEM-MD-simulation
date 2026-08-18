"""Render a gsd.hoomd trajectory as an animated GIF using fresnel.

Requires Pillow (PIL) in addition to fresnel/gsd/numpy - not yet declared in
any requirements file for this repo.
"""

import fresnel
import gsd.hoomd
import numpy as np
from PIL import Image


def _default_type_style():
    return {
        "Strep_cent": {"radius": 1.0, "color": (0.9, 0.6, 0.1)},  # node hub
        "Strep_cons": {"radius": 0.5, "color": (0.2, 0.4, 0.9)},  # binding arms
        "Biotin": {"radius": 0.4, "color": (0.2, 0.8, 0.3)},
    }


_FALLBACK_STYLE = {"radius": 0.5, "color": (0.6, 0.6, 0.6)}


def _resolve_style_arrays(frame, type_style):
    type_names = frame.particles.types
    typeid = frame.particles.typeid
    radii = np.empty(len(typeid), dtype=float)
    colors = np.empty((len(typeid), 3), dtype=float)
    seen_unstyled = set()
    for i, tid in enumerate(typeid):
        name = type_names[tid]
        style = type_style.get(name)
        if style is None:
            if name not in seen_unstyled:
                print(f"rendering: no style for particle type {name!r}, using fallback")
                seen_unstyled.add(name)
            style = _FALLBACK_STYLE
        radii[i] = style["radius"]
        colors[i] = style["color"]
    return radii, colors


def _fixed_camera(scene, box, padding):
    # temporarily add the box to the scene purely so Orthographic.fit() has
    # something to fit around; the caller re-adds a permanent box afterward
    # if it wants one drawn.
    probe_box = fresnel.geometry.Box(scene, box, box_radius=0.0)
    camera = fresnel.camera.Orthographic.fit(scene, margin=padding - 1.0)
    probe_box.remove()
    return camera


def _frame_to_pil_image(tracer):
    return Image.fromarray(tracer.output[:], mode="RGBA")


def _save_gif(images, filename, fps):
    duration_ms = int(1000 / fps)
    frames_rgb = [im.convert("RGB").quantize(colors=256) for im in images]
    frames_rgb[0].save(
        filename,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=duration_ms,
        loop=0,
    )


def render_movie(
    frames,
    filename="movie.gif",
    resolution=(500, 500),
    fps=5,
    samples=32,
    type_style=None,
    draw_box=True,
    box_radius=0.05,
    camera_padding=1.15,
    device=None,
):
    """Render a gsd.hoomd trajectory (or any indexable sequence of Frames) to an animated GIF.

    `samples` is the main quality/speed dial for the path tracer - higher
    values reduce noise at the cost of render time per frame.
    """
    if type_style is None:
        type_style = _default_type_style()

    device = device or fresnel.Device(mode="auto")
    width, height = resolution
    tracer = fresnel.tracer.Path(device, width, height)

    box = frames[0].configuration.box

    scene = fresnel.Scene(device=device)
    scene.lights = fresnel.light.lightbox()
    scene.camera = _fixed_camera(scene, box, camera_padding)

    if draw_box:
        fresnel.geometry.Box(scene, box, box_radius=box_radius)

    material = fresnel.material.Material(roughness=0.8, specular=0.5, primitive_color_mix=1.0)

    sphere_geometry = None
    images = []
    for frame in frames:
        if sphere_geometry is not None:
            sphere_geometry.remove()

        radii, colors = _resolve_style_arrays(frame, type_style)
        sphere_geometry = fresnel.geometry.Sphere(
            scene, position=frame.particles.position, radius=radii
        )
        sphere_geometry.color[:] = fresnel.color.linear(colors)
        sphere_geometry.material = material

        tracer.sample(scene, samples=samples)
        images.append(_frame_to_pil_image(tracer))

    _save_gif(images, filename, fps)
    return filename
