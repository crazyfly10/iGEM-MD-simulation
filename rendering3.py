import fresnel
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 02-Advanced-topics/01-Devices: create the Device explicitly rather than
# relying on Scene's implicit default. fresnel.Device() auto-selects GPU
# ray tracing if available, otherwise CPU.
device = fresnel.Device()

# 02-Advanced-topics/05-GSD-visualization: open the trajectory and read a
# single frame (the first one) to render.
with gsd.hoomd.open(name="trajectory.gsd", mode="r") as gsd_file:
    snap = gsd_file[0]

box = snap.configuration.box

# Color by typeid, following the GSD-visualization tutorial's pattern -
# Strep_cent red, Strep_cons green, Biotin blue.
N = snap.particles.N
particle_types = snap.particles.typeid
type_names = snap.particles.types
colors = np.empty((N, 3))
colors[particle_types == type_names.index("Strep_cent")] = fresnel.color.linear([0.863, 0.157, 0.157])
colors[particle_types == type_names.index("Strep_cons")] = fresnel.color.linear([0.000, 0.698, 0.478])
colors[particle_types == type_names.index("Biotin")] = fresnel.color.linear([0.588, 0.294, 0.118])

scene = fresnel.Scene(device=device)

# Spheres for every particle in the system.
geometry = fresnel.geometry.Sphere(scene, N=N)
#geometry.position[:] = snap.particles.position
# 00-Basic-tutorials/02-Material-properties + 03-Outline-materials: give the
# spheres a material and a dark outline so individual particles read clearly.
geometry.material = fresnel.material.Material(roughness=0.9)
geometry.outline_width = 0.05
# use per-particle color instead of a single material.color
geometry.material.primitive_color_mix = 1.0
geometry.color[:] = fresnel.color.linear(colors)

# Per-particle radii, matching the physical sizes actually used in
# sample_strep-biotin_sim.py rather than one size for every type
# (geometry.radius is a settable (N,) array, same as .position/.color).
radius = np.empty(N)
# Strep_cons: particle_radius = 1.7 there - used for the rigid body's
# moment-of-inertia calc and for sizing the Strep_cons-Strep_cons
# excluded-volume repulsion (contact at r = 2 * particle_radius).
radius[particle_types == type_names.index("Strep_cons")] = 1.7
# Biotin: sigma_biotin = 0.5 there is the Biotin-Biotin excluded-volume
# diameter, so radius = sigma_biotin / 2.
radius[particle_types == type_names.index("Biotin")] = 0.5 / 2
# Strep_cent: the rigid body's virtual center particle. sample_strep-
# biotin_sim.py never gives it a physical radius - its translational mass
# is a placeholder (1.0) and its moment of inertia comes from the
# constituent geometry, not a self-radius - so this is just a small marker
# for the body center, not a real steric size.
radius[particle_types == type_names.index("Strep_cent")] = 0.05
geometry.radius[:] = radius

# create box in fresnel
fresnel.geometry.Box(scene, box, box_radius=0.02)

# 00-Basic-tutorials/04-Scene-properties: solid white background.
scene.background_alpha = 1.0
scene.background_color = fresnel.color.linear([1, 1, 1])

# 00-Basic-tutorials/04-Scene-properties + 05-Lighting-setups: camera fit to
# the scene, ring lighting.
scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.lightbox()

# 02-Advanced-topics/02-Tracer-methods: use the Path tracer directly (rather
# than the fresnel.pathtrace() convenience function) - it supports soft
# lighting/reflections and averages many samples together via .sample().
tracer = fresnel.tracer.Path(device=device, w=1200, h=1200)
tracer.sample(scene, samples=64, light_samples=40)

# 02-Advanced-topics/04-Rendering-images-in-matplotlib: [:] converts the
# tracer's output buffer to a plain numpy RGBA array for imshow (interactive
# display only - the actual PNG file is written via PIL below).
#image = tracer.output[:]

vid = []
for frame in gsd_file:
    geometry.position[:] = frame.particles.position
    frame_arr = Image.fromarray(tracer.output[:], mode="RGBA")
    vid.append(frame_arr)

vid[0].save(
    'sim.gif', 
    save_all = "True", 
    append_images = vid[1:],
    duration = 10,
)

print('saved sim.gif')
'''
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, interpolation="lanczos")
ax.set_xticks([])
ax.set_yticks([])
'''
'''
PIL.Image.fromarray(image, mode="RGBA").save("frame.png")
print("saved frame.png")
'''