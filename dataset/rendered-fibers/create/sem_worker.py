"""This module is intended to run with the following command:
docker run --rm -i -u $(id -u):$(id -g) --volume "$(pwd):/kubric" kubruntu_sdf /usr/bin/python3 'create/sem_worker.py' <task_id>

It creates a single scene and stores 3 files. An rgba, a segmentation with 
the first 10 fibers and a json with the information about those 10 fibers.
"""
import shutil
from sdf import capped_cylinder, Y, X, ease

import logging
import kubric as kb
from kubric.assets.asset_source import AssetSource
from kubric.core.color import Color
from kubric.core.materials import FlatMaterial, PrincipledBSDFMaterial
from kubric.core.traits import RGBA
from kubric.renderer.blender import Blender as KubricRenderer
import os
from uuid import uuid1
import numpy as np
import json
import sys
from itertools import combinations
from random import choice

logging.basicConfig(level="WARNING", filename='output/output.log')
scene_uid = sys.argv[1]
temp_folder = os.path.join("temp", scene_uid)
os.makedirs(temp_folder, exist_ok=True)


def create_fiber(p1, p2, diameter=0.1):
    uid = str(uuid1())
    filepath = os.path.join(temp_folder, f"{uid}.obj")
    f = capped_cylinder(p1, p2, diameter / 2)
    # f = f.bend_linear(-Y, 1.5*Y, X/2, ease.in_out_quad)
    f.save(filepath, sparse=False, samples=2**18, verbose=False)
    obj = kb.FileBasedObject(
        asset_id=uid,
        render_filename=filepath,
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None,
    )
    return obj


def rectangle_perimeter(p1, p2, t):
    """t is in range [0,1]
    points are here:
    ---2
    |  |
    1---
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    d = p2 - p1

    if 0 <= t < 0.25:
        return np.array([4 * t * d[0] + p1[0], p1[1]])
    elif 0.25 <= t < 0.5:
        return np.array([p2[0], 4 * (t - 0.25) * d[1] + p1[1]])
    elif 0.5 <= t < 0.75:
        return np.array([p2[0] - 4 * (t - 0.5) * d[0], p2[1]])
    elif 0.75 <= t <= 1:
        return np.array([p1[0], p2[1] - 4 * (t - 0.75) * d[1]])


def random_line_in_perimeter(p1, p2):
    sides = [[0.00, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.00]]
    comb = [x for x in combinations(sides, 2)]
    s1, s2 = choice(comb)
    t1 = np.random.uniform(*s1)
    t2 = np.random.uniform(*s2)
    return rectangle_perimeter(p1, p2, t1), rectangle_perimeter(p1, p2, t2)


def random_fiber(z, d):
    p1, p2 = random_line_in_perimeter([-0.6, -0.6], [0.6, 0.6])
    p1 = [*p1, z]
    p2 = [*p2, z]
    return create_fiber(p1, p2, d), p1, p2

# bg_color
c = np.random.rand()
color_bg = Color(c, c, c, 0.5)
bg_material = PrincipledBSDFMaterial(
  color=color_bg
)

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(256, 256), background = Color(c, c, c, 1))
renderer = KubricRenderer(scene, use_denoising=False)

c = np.random.rand()
color_fg = Color(c, c, c, 1)
fg_material = PrincipledBSDFMaterial(
  color=color_fg,
  roughness=np.random.rand()
)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -1), material=bg_material, background=True)
mu = np.random.uniform(0.03, 0.15)
d = np.random.normal(mu, 0.01, 50)


fg_material = PrincipledBSDFMaterial(
      color=color_fg,
      metallic = np.random.rand(),
      specular = np.random.rand(),
      specular_tint = np.random.rand(),
      roughness = np.random.rand()
    )

last_z = 0.1

task_data = {}
for i in range(10):
    last_z = last_z - d[i]*0.5
    obj, p1, p2 = random_fiber(last_z, d[i])
    obj.material = fg_material
    scene += obj
    task_data[i] = {'d':d[i], 'p1':p1, 'p2': p2}
scene += kb.DirectionalLight(
    name="sun", position=np.random.rand(3)*3, look_at=(0, 0, 0), intensity=np.random.uniform(0.5,1.5)
)
scene.camera = kb.OrthographicCamera(position=(0, 0, 3), orthographic_scale=1)
frame = renderer.render_still(return_layers=['segmentation'])
kb.write_palette_png(frame["segmentation"], f"output/{scene_uid}_seg10.png")

for i in range(10, 40):
    last_z = last_z - d[i]*0.5
    if last_z < -1.0:
      break
    obj, p1, p2 = random_fiber(last_z, d[i])
    obj.material = fg_material
    scene += obj
    task_data[i] = {'d':d[i], 'p1':p1, 'p2': p2}

with open(f'output/{scene_uid}.json', 'w') as file:
    json.dump(task_data, file)
# renderer.save_state("output/helloworld.blend")

frame = renderer.render_still(return_layers=['rgba', 'segmentation'])
# --- save the output as pngs
kb.write_png(frame["rgba"], f"output/{scene_uid}.png")
kb.write_palette_png(frame["segmentation"], f"output/{scene_uid}_seg.png")
np.savez(f'output/{scene_uid}_seg.npz',y=frame['segmentation'])

shutil.rmtree(temp_folder)