#!/usr/bin/env python3
"""Bottom row for the 2x3 composites, driven by saved value fields.

Column 1: vehicles under the levelsetpy benchmark (reference field).
Column 2: vehicles under HJ-Gauss (full field).
Column 3: 2D chart of | ||x_ref(t)|| - ||x_hj(t)|| | per agent, with the
          mean and maximum over agents.
Each vehicle is the Fig. 4 shuttle drawing from levtoms.pdf as a textured
decal. Rates come from grad v of the matching field (fieldcontrol.py).
"""
import argparse, io
from pathlib import Path

import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from hjgauss.render import fieldcontrol as F

CELL_W, CELL_H = 300, 342
SUPER = 2
ASSETS = Path(__file__).resolve().parent / 'assets'   # written by hjgauss.render.decals
DECAL_FILES = {'rocket_pursuer_key.png': ASSETS / 'rocket_pursuer_blue_key.png',
               'rocket_evader_key.png': ASSETS / 'rocket_evader_red_key.png'}
SIGN = {'rockets': -1.0, 'dubins_relative': 1.0, 'murmuration': 1.0}
FIELDS = {  # (reference archive, full archive)
    'rockets': ('rockets_reference_160_400_L6_c0.2', 'rockets_full_160_400_L6_c0.2'),
    'dubins_relative': ('dubins_relative_reference_160_400_L6_c0.2',
                        'dubins_relative_full_160_400_L6_c0.2'),
    'murmuration': ('murmuration_reference_160_400_L6_c0.2',
                    'murmuration_full_160_400_L6_c0.2'),
}
ALT = 0.6  # quadrotor altitude above the ground (murmuration only)
STARTS = {
    'rockets': np.array([[-1.566, 1.974, -0.169]]),   # one pursuer, captured near t = 0.36
    'dubins_relative': np.array([[0.322, -0.371, 0.372]]),  # stays outside the radius-0.12 capture set (near miss)  # inside the BRT at T = 1; both fields give v < 0 and the controls capture
    'murmuration': np.array([[0.900, 0.000, -3.142], [0.561, 0.704, -2.244], [-0.200, 0.877, -1.346], [-0.811, 0.390, -0.449], [-0.811, -0.390, 0.449], [-0.200, -0.877, 1.346], [0.561, -0.704, 2.244]]),   # cordon: seven drones, one per disk of the seven-predator ring
}


def dubins_world(X, we, dt, speed=1.0, start=(-0.4, 0.0)):
    """Relative Dubins states -> world poses, for the evader (at speed 1) and the pursuer.

    The relative frame is attached to the evader. The evader heading h integrates its
    steering w_e; the evader moves at unit speed along h; the pursuer sits at the
    relative offset rotated into the world by h, with world heading h + x3.
    Returns (K, 2, 3): row 0 pursuer (x, y, heading), row 1 evader (x, y, heading).
    """
    K = X.shape[0]
    h = np.concatenate([[0.0], np.cumsum(we[:, 0]) * dt])
    E = np.zeros((K, 2))
    E[1:] = np.cumsum(np.c_[np.cos(h[:-1]), np.sin(h[:-1])] * speed * dt, 0)
    E += np.asarray(start, float)
    c, sn = np.cos(h), np.sin(h)
    px = E[:, 0] + c * X[:, 0, 0] - sn * X[:, 0, 1]
    py = E[:, 1] + sn * X[:, 0, 0] + c * X[:, 0, 1]
    world = np.zeros((K, 2, 3))
    world[:, 0] = np.c_[px, py, h + X[:, 0, 2]]
    world[:, 1] = np.c_[E[:, 0], E[:, 1], h]
    return world


def rockets_world(X, ue, dt, speed=1.0, start=(0.0, 0.0)):
    """Relative rocket states -> world poses in the x-z plane, for both rockets.

    The evader steers with rate ue and flies at speed 1 along its heading; the pursuer
    sits at the relative offset rotated into the world by the evader heading.
    Returns (K, 2, 3): row 0 pursuer (x, z, heading), row 1 evader (x, z, heading).
    """
    K = X.shape[0]
    h = np.concatenate([[0.0], np.cumsum(ue[:, 0]) * dt])
    E = np.zeros((K, 2))
    E[1:] = np.cumsum(np.c_[np.cos(h[:-1]), np.sin(h[:-1])] * speed * dt, 0)
    E += np.asarray(start, float)
    c, sn = np.cos(h), np.sin(h)
    px = E[:, 0] + c * X[:, 0, 0] - sn * X[:, 0, 1]
    pz = E[:, 1] + sn * X[:, 0, 0] + c * X[:, 0, 1]
    world = np.zeros((K, 2, 3))
    world[:, 0] = np.c_[px, pz, h + X[:, 0, 2]]
    world[:, 1] = np.c_[E[:, 0], E[:, 1], h]
    return world


def rollouts(name, hj_root, T, dt):
    ref_key, hj_key = FIELDS[name]
    out = {}
    for tag, key in [('ref', ref_key), ('hj', hj_key)]:
        f = F.load_field(Path(hj_root) / (key + '.npz'))
        trace = []
        if name == 'rockets':   # evader steers against its gradient, pursuer closes in (chase convention)
            t, X = F.rollout(f, STARTS[name], T, dt, F.rockets_controls_evade, F.rockets_rhs, sgn=SIGN[name], trace=trace)
        else:  # relative Dubins and murmuration share one Hamiltonian
            ctrl = F.dubins_controls_swapped if name == 'dubins_relative' else F.dubins_controls
            t, X = F.rollout(f, STARTS[name], T, dt, ctrl, F.dubins_rhs, sgn=SIGN[name], trace=trace)
        out[tag] = X
        out['t'] = t
        if name == 'murmuration':   # place each flock's drones at its own centre in the plane
            W = X.copy()
            per = MURMUR_NPRED['n']
            for i in range(X.shape[1]):
                W[:, i, 0] += MURMUR_CENTRES['c'][i // per, 0]
                W[:, i, 1] += MURMUR_CENTRES['c'][i // per, 1]
            out[tag + '_world'] = W
        if name == 'dubins_relative':
            we = np.array([u[0] for u in trace])            # evader steering, (steps, 1)
            out[tag + '_world'] = dubins_world(X, we, dt)
        if name == 'rockets':
            ue = np.array([u[0] for u in trace])            # evader steering rate
            out[tag + '_world'] = rockets_world(X, ue, dt)
    return out


def quad_xml(i, S=2.6):
    """Quadrotor (X frame) as a mocap body: arms, four translucent rotors, LEDs."""
    rot = ''.join(
        f'<geom type="cylinder" pos="{sx*0.17*S:.3f} {sy*0.17*S:.3f} {0.05*S:.3f}" size="{0.095*S:.3f} {0.004*S:.3f}" rgba="0.85 0.88 0.95 0.35" contype="0" conaffinity="0"/>'
        f'<geom type="sphere" pos="{sx*0.17*S:.3f} {sy*0.17*S:.3f} {0.03*S:.3f}" size="{0.02*S:.3f}" rgba="{"1 0.2 0.2 1" if sx > 0 else "0.2 1 0.3 1"}" contype="0" conaffinity="0"/>'
        for sx in (-1, 1) for sy in (-1, 1))
    a = 0.19 * S
    return (f'<body name="v{i}" mocap="true">'
            f'<geom type="box" size="{0.05*S:.3f} {0.05*S:.3f} {0.02*S:.3f}" rgba="0.12 0.12 0.16 1" contype="0" conaffinity="0"/>'
            f'<geom type="capsule" fromto="-{a:.3f} -{a:.3f} 0 {a:.3f} {a:.3f} 0" size="{0.012*S:.3f}" rgba="0.2 0.2 0.25 1" contype="0" conaffinity="0"/>'
            f'<geom type="capsule" fromto="-{a:.3f} {a:.3f} 0 {a:.3f} -{a:.3f} 0" size="{0.012*S:.3f}" rgba="0.2 0.2 0.25 1" contype="0" conaffinity="0"/>'
            + rot + '</body>')


DANGER_R = 0.12   # capture radius of the Dubins danger zone (smaller than the v = 0 contour)
CAPTURE_R = {'dubins_relative': 0.06, 'rockets': 1.5}   # Dubins: reach of an IED (IED plus car half-width)


# IED centres in the relative plane, spread out but at least 0.17 from the evader's path
# (found by search; the path comes from the field-driven controls, so it is fixed)
IED_POINTS = [(0.758, -0.780), (-0.676, -0.752), (-0.099, -0.692), (-0.783, 0.739), (0.649, 0.469)]


def danger_xml():
    """Danger zone: a red ring with IED-style markers (mine body + warning ring)."""
    out = []
    for x, y in IED_POINTS:
        out.append(f'<geom type="cylinder" pos="{x:.4f} {y:.4f} 0.012" size="0.022 0.012" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>')
        out.append(f'<geom type="cylinder" pos="{x:.4f} {y:.4f} 0.003" size="0.055 0.002" rgba="1 0.15 0.1 0.9" contype="0" conaffinity="0"/>')
    return ''.join(out)


CAR_SCALE = 0.25 / 462.0   # car length 0.25 units; mesh body length is 462 cm


def car_xml(i, rgba):
    """Prius (osrf car_demo) as a mocap body, body mesh colored by role."""
    s = CAR_SCALE
    wheels = ''.join(
        f'<geom type="mesh" mesh="wheel" pos="{sx*75*s:.6f} {sy*150*s:.6f} {30*s:.6f}" rgba="0.08 0.08 0.09 1" contype="0" conaffinity="0"/>'
        for sx in (-1, 1) for sy in (-1, 1))
    return (f'<body name="v{i}" mocap="true">'
            f'<geom type="mesh" mesh="body" rgba="{rgba}" contype="0" conaffinity="0"/>'
            + wheels + '</body>')


def scene(name, n_agents, gz):
    half = DECAL_HALF[name]
    if name == 'dubins_relative':
        cars = ''.join(car_xml(i, '0.85 0.12 0.12 1') for i in range(n_agents))
        cars += car_xml(n_agents, '0.12 0.30 0.85 1')   # evader, blue
        cars += danger_xml()                             # IEDs are fixed in the world
        return f"""<mujoco>
  <visual><headlight ambient=".55 .55 .58" diffuse=".75 .75 .75"/><rgba haze="0.72 0.80 0.90 1"/>
    <quality shadowsize="4096" offsamples="8"/><global offwidth="1280" offheight="1280"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".58 .74 .96" rgb2=".07 .09 .20" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".36 .42 .33" rgb2=".27 .33 .25" width="512" height="512" mark="edge" markrgb=".82 .84 .80"/>
    <material name="ground" texture="grid" texrepeat="10 10" reflectance=".18" specular=".25" shininess=".15"/>
    <mesh name="body" file="prius/hybrid_body.obj" scale="{CAR_SCALE} {CAR_SCALE} {CAR_SCALE}"/>
    <mesh name="wheel" file="prius/wheel.obj" scale="{CAR_SCALE} {CAR_SCALE} {CAR_SCALE}"/>
  </asset>
  <worldbody>
    <light directional="true" pos="4 -6 9" dir="-.35 .55 -1" diffuse=".95 .88 .78" castshadow="false"/>
    <geom name="ground" type="plane" pos="0 0 {gz}" size="16 16 .1" material="ground"/>
    {cars}
  </worldbody>
</mujoco>"""
    if name == 'murmuration':
        return f"""<mujoco>
  <visual><headlight ambient=".55 .55 .58" diffuse=".75 .75 .75"/><rgba haze="0.72 0.80 0.90 1"/>
    <quality shadowsize="4096" offsamples="8"/><global offwidth="1280" offheight="1280"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".58 .74 .96" rgb2=".07 .09 .20" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".36 .42 .33" rgb2=".27 .33 .25" width="512" height="512" mark="edge" markrgb=".82 .84 .80"/>
    <material name="ground" texture="grid" texrepeat="10 10" reflectance=".18" specular=".25" shininess=".15"/>
  </asset>
  <worldbody>
    <light directional="true" pos="4 -6 9" dir="-.35 .55 -1" diffuse=".95 .88 .78" castshadow="false"/>
    <geom name="ground" type="plane" pos="0 0 {gz}" size="16 16 .1" material="ground"/>
    {'' if BIRD_MODEL['kind'] == 'pigeon' else ''.join(quad_xml(i) for i in range(n_agents))}
  </worldbody>
</mujoco>"""
    if name == 'rockets':   # rockets are drawn as 3D geometry per frame; placeholders stay invisible
        decals = ''.join(
            f'<body name="v{i}" mocap="true"><geom type="plane" size="0.01 0.01 1" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>'
            for i in range(n_agents + 1))
    else:
        decals = ''.join(
            f'<body name="v{i}" mocap="true"><geom type="plane" size="{half} {half} 1" material="decal_p" contype="0" conaffinity="0"/></body>'
            for i in range(n_agents))
    if name != 'rockets':
        decals += f'<body name="v{n_agents}" mocap="true"><geom type="plane" size="{half} {half} 1" material="decal_e" contype="0" conaffinity="0"/></body>'
    terrain = rocket_terrain_xml() if name == 'rockets' else ''
    ground_mat = 'sea' if name == 'rockets' else 'ground'
    sky = ''
    return f"""<mujoco>
  <visual><headlight ambient=".55 .55 .58" diffuse=".75 .75 .75"/><rgba haze="0.72 0.80 0.90 1"/>
    <quality shadowsize="4096" offsamples="8"/><global offwidth="1280" offheight="1280"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".58 .74 .96" rgb2=".07 .09 .20" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".36 .42 .33" rgb2=".27 .33 .25" width="512" height="512" mark="edge" markrgb=".82 .84 .80"/>
    <material name="ground" texture="grid" texrepeat="10 10" reflectance=".18" specular=".25" shininess=".15"/>
    <texture name="pursuer" type="2d" file="rocket_pursuer_key.png"/>
    <texture name="evader" type="2d" file="rocket_evader_key.png"/>
    <material name="decal_p" texture="pursuer" rgba="1 1 1 1" specular=".3" shininess=".2"/>
    <material name="sea" rgba=".10 .32 .42 1" specular=".8" shininess=".6" reflectance=".35"/>
    <material name="cloud" rgba=".97 .97 .98 1" specular=".1"/>
    <material name="grass" rgba=".22 .58 .24 1" specular=".05"/>
    <material name="frond" rgba=".12 .48 .18 1" specular=".05"/>
    <material name="trunk" rgba=".42 .28 .14 1" specular=".05"/>
    <material name="sand" rgba=".80 .64 .42 1" specular=".05" shininess=".05"/>
    <material name="rock" rgba=".45 .30 .22 1" specular=".1"/>
    <material name="mesa" rgba=".62 .40 .26 1" specular=".1"/>
    <material name="steel" rgba=".55 .58 .62 1" specular=".5" shininess=".5"/>
    <material name="decal_e" texture="evader" rgba="1 1 1 1" specular=".3" shininess=".2"/>
  </asset>
  <worldbody>
    <light directional="true" pos="4 -6 9" dir="-.35 .55 -1" diffuse=".95 .88 .78" castshadow="false"/>
    <geom name="ground" type="plane" pos="0 0 {gz}" size="16 16 .1" material="{ground_mat}"/>
    {terrain}
    {decals}
  </worldbody>
</mujoco>"""


# oasis islands (x, y, radius) in the rockets scene; visible in the side view
ISLANDS = [(-4.5, 2.0, 1.0), (4.0, 3.0, 0.8), (0.0, 4.5, 1.3), (-7.5, 3.5, 0.7)]


def rocket_terrain_xml():
    """Ocean scene for the rockets: green oasis islands with palms, and cloud banks."""
    rng = np.random.default_rng(21)
    out = []
    islands = ISLANDS
    for x, y, r in islands:
        z0 = -3.4
        out.append(f'<geom type="ellipsoid" pos="{x} {y} {z0 + 0.05:.2f}" size="{r} {0.8 * r:.2f} 0.12" material="sand" contype="0" conaffinity="0"/>')
        out.append(f'<geom type="ellipsoid" pos="{x} {y} {z0 + 0.12:.2f}" size="{0.85 * r:.2f} {0.7 * r:.2f} 0.15" material="grass" contype="0" conaffinity="0"/>')
        for _ in range(int(3 + r)):
            px = x + rng.uniform(-0.5, 0.5) * r; py = y + rng.uniform(-0.4, 0.4) * r
            out.append(f'<geom type="capsule" fromto="{px:.2f} {py:.2f} {z0 + 0.1:.2f} {px:.2f} {py:.2f} {z0 + 0.9:.2f}" size="0.04" material="trunk" contype="0" conaffinity="0"/>')
            out.append(f'<geom type="ellipsoid" pos="{px:.2f} {py:.2f} {z0 + 1.0:.2f}" size="0.32 0.32 0.16" material="frond" contype="0" conaffinity="0"/>')
    for _ in range(14):
        cx, cy = rng.uniform(-10, 10), rng.uniform(-10, 12)
        cz = rng.uniform(3.0, 5.5)
        out.append(f'<geom type="ellipsoid" pos="{cx:.2f} {cy:.2f} {cz:.2f}" size="{rng.uniform(1.2, 2.6):.2f} {rng.uniform(0.8, 1.6):.2f} 0.35" material="cloud" contype="0" conaffinity="0"/>')
    return ''.join(out)


def place(model, data, name, X_k, n):
    """Set mocap decals for agents (rows of X_k) and the origin vehicle."""
    if name == 'rockets':   # rows are world poses (x, z, heading): pursuer(s), then evader
        for i in range(len(X_k)):
            x, z, h = X_k[i]
            body = model.body(f'v{i}').mocapid[0]
            data.mocap_pos[body] = [x, 0.0, z]
            q = np.zeros(4); q2 = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array([0, 1.0, 0]), np.pi / 2 - h)
            mujoco.mju_axisAngle2Quat(q2, np.array([1.0, 0, 0]), np.pi / 2)
            qout = np.zeros(4); mujoco.mju_mulQuat(qout, q, q2)
            data.mocap_quat[body] = qout
        return
    if name == 'dubins_relative':
        n = len(X_k) - 1          # rows are the pursuers followed by the evader
        for i in range(n):
            x, y, th = X_k[i]  # one pursuer
            body = model.body(f'v{i}').mocapid[0]
            data.mocap_pos[body] = [x, y, 0.0]
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), th - np.pi / 2)
            data.mocap_quat[body] = q
        ex, ey, eh = X_k[n]   # evader: world pose from the trajectory
        body = model.body(f'v{n}').mocapid[0]
        data.mocap_pos[body] = [ex, ey, 0.0]
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), eh - np.pi / 2)
        data.mocap_quat[body] = q
        return
    if name == 'murmuration' and BIRD_MODEL['kind'] == 'pigeon':
        return   # pigeons are drawn per frame, not as mocap bodies
    if name == 'murmuration':
        for i in range(n):
            x, y, th = X_k[i]
            body = model.body(f'v{i}').mocapid[0]
            data.mocap_pos[body] = [x, y, ALT]
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), th)
            data.mocap_quat[body] = q
        return
    for i in range(n):
        x1, x2, th = X_k[i]
        if name == 'rockets':
            body = model.body(f'v{i}').mocapid[0]
            data.mocap_pos[body] = [x1, 0.0, x2]
            alpha = np.pi / 2 - th
            # plane normal -> -y, then in-plane rotation so image up = (cos th, 0, sin th)
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array([0, 1.0, 0]), alpha)
            q2 = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q2, np.array([1.0, 0, 0]), np.pi / 2)
            qout = np.zeros(4); mujoco.mju_mulQuat(qout, q, q2)
            data.mocap_quat[body] = qout
        else:
            body = model.body(f'v{i}').mocapid[0]
            data.mocap_pos[body] = [x1, x2, 0.0]
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), th - np.pi / 2)
            data.mocap_quat[body] = q
    body = model.body(f'v{n}').mocapid[0]
    if name == 'rockets':
        data.mocap_pos[body] = [0, 0, 0]
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.pi / 2)
        data.mocap_quat[body] = q
    else:
        data.mocap_pos[body] = [0, 0, 0]
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), -np.pi / 2)
        data.mocap_quat[body] = q


def _trail_pts(name, hist):
    z = np.zeros_like(hist[..., 0])
    if name == 'rockets':
        return np.stack([hist[..., 0], z, hist[..., 1]], -1)
    if name == 'murmuration':
        return np.stack([hist[..., 0], hist[..., 1], z + ALT], -1)
    return np.stack([hist[..., 0], hist[..., 1], z], -1)


def add_explosion(scn, centre, age):
    """Fireball and rising smoke at the evader after capture; age in elapsed time units."""
    rng = np.random.default_rng(3)
    grow = min(age / 0.35, 1.0)
    id3 = np.eye(3).ravel()
    for k in range(22):   # fire: orange-yellow core spreading outward
        off = rng.normal(0, 0.008 + 0.02 * grow, 3); off[2] = abs(off[2]) + 0.01
        rad = (0.025 + 0.05 * grow) * (0.6 + 0.8 * rng.random())
        g = scn.geoms[scn.ngeom]; scn.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]),
                            np.asarray(centre, float) + off, id3,
                            np.array([1.0, 0.45 + 0.45 * rng.random(), 0.05, 0.85]))
        g.emission = 1.0
    for k in range(10):   # smoke: dark gray, rising and widening
        rise = 0.05 + 0.15 * max(age, 0.0)
        off = rng.normal(0, 0.01 + 0.02 * grow, 3); off[2] = rise + abs(off[2])
        rad = 0.03 + 0.07 * max(age, 0.0) * (0.5 + rng.random())
        g = scn.geoms[scn.ngeom]; scn.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]),
                            np.asarray(centre, float) + off, id3, np.array([0.18, 0.17, 0.16, 0.55]))


WAVE_T = {'t': 0.0}   # elapsed time, set per frame, drives the sea motion


def frame_axes(d):
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    a = np.array([1., 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1., 0])
    x = np.cross(a, d); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(d, x), d]).ravel()


def add_geom(scn, kind, size, pos, mat, rgba, emission=0.0, spec=0.3):
    g = scn.geoms[scn.ngeom]; scn.ngeom += 1
    sz = np.zeros(3); sz[:len(size)] = size
    mujoco.mjv_initGeom(g, kind, sz, np.asarray(pos, float), np.asarray(mat, float), np.asarray(rgba, float))
    g.emission = emission; g.specular = spec; g.shininess = 0.5


def add_rocket(scn, pos, heading, rgb, thrust, t):
    """Solid rocket: white body, coloured nose, fins, nozzle, and a flickering nitrogen plume."""
    d = np.array([np.cos(heading), 0.0, np.sin(heading)])   # thrust/nose direction in the x-z plane
    R = frame_axes(d)
    p = np.asarray(pos, float)
    body_c = p + d * 0.30
    add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.055, 0.30], body_c, R, [0.92, 0.93, 0.96, 1.0], spec=0.7)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.055, 0.055, 0.11], p + d * 0.66, R, [*rgb, 1.0], spec=0.8)
    for k in range(4):
        ang = k * np.pi / 2
        side = np.array([np.cos(heading + np.pi / 2) * 0, 0, 0])
        off = np.array([np.cos(ang) * 0.06, np.sin(ang) * 0.06, 0.0])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.012, 0.09, 0.06], p - d * 0.22 + off,
                 frame_axes(d), [*rgb, 1.0], spec=0.4)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.05, 0.05], p - d * 0.26, R, [0.15, 0.15, 0.17, 1.0], spec=0.2)
    if thrust > 0:
        flick = 1.0 + 0.10 * np.sin(31 * t + 2 * heading)
        base = p - d * 0.30
        for k, (ln, rad, col) in enumerate([(0.55, 0.11, [1.0, 0.45, 0.10, 0.45]),
                                             (0.36, 0.07, [1.0, 0.75, 0.25, 0.70]),
                                             (0.18, 0.04, [1.0, 0.97, 0.80, 0.90])]):
            add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [rad, rad, ln * thrust * flick],
                     base - d * (ln * thrust * flick), R, col, emission=1.0, spec=0.0)
        for k in range(6):   # nitrogen puffs trailing the plume
            off = np.array([np.sin(k * 2.1 + t * 3) * 0.04, 0, np.cos(k * 1.3) * 0.03])
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05 + 0.03 * k], base - d * (0.45 + 0.12 * k) + off,
                     np.eye(3).ravel(), [0.85, 0.87, 0.90, 0.22 - 0.02 * k], spec=0.0)


def add_sea_crests(scn, t, n=70, z=-3.4):
    """Crests drift across the sea, so the ocean reads as flowing."""
    for k in range(n):
        x = ((k * 0.73 + t * 1.6) % 26.0) - 13.0
        y = -2.0 + ((k * 0.37) % 4.0)
        length = 0.9 + 0.5 * ((k * 0.61) % 1.0)
        add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.012, length], [x, y, z + 0.01],
                 frame_axes([1.0, 0.0, 0.0]), [0.85, 0.95, 1.0, 0.35], spec=0.6)


FLOCK_COLOURS = [((0.36, 0.56, 0.62), (0.48, 0.62, 0.70)),   # flock 1: blue-grey
                 ((0.80, 0.42, 0.14), (0.92, 0.58, 0.30)),    # flock 2: orange
                 ((0.52, 0.26, 0.70), (0.66, 0.44, 0.84))]    # flock 3: violet


def add_pigeon(scn, pos, heading, t, phase, scale=6.5, flock=0):
    """Pigeon: body, head, tail, and two wings that beat in time; colour set by flock."""
    body_rgb, wing_rgb = FLOCK_COLOURS[flock % len(FLOCK_COLOURS)]
    R = np.column_stack([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    p = np.asarray(pos, float)
    body = [*body_rgb, 1.0]
    add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.10 * scale / 2.4, 0.040 * scale / 2.4, 0.034 * scale / 2.4],
             p, R.ravel(), body, spec=0.5)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.030 * scale / 2.4], p + R @ [0.10 * scale / 2.4, 0, 0.012],
             R.ravel(), [0.42, 0.46, 0.52, 1.0], spec=0.5)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.035 * scale / 2.4, 0.022 * scale / 2.4, 0.006 * scale / 2.4],
             p + R @ [-0.10 * scale / 2.4, 0, 0.0], R.ravel(), [0.30, 0.34, 0.40, 1.0], spec=0.2)
    flap = 0.7 * np.sin(2 * np.pi * 5.0 * t + phase)
    for sy in (-1, 1):
        Rw = R @ Rx_np(sy * flap)
        centre = p + R @ [0.0, sy * 0.06 * scale / 2.4, 0.01]
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.07 * scale / 2.4, 0.075 * scale / 2.4, 0.006 * scale / 2.4],
                 centre, Rw.ravel(), [*wing_rgb, 0.95], spec=0.3)


def Rx_np(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _shot(name, X_tk, model, renderer, cam_spec, hide, hist=None, boom=None):
    data = mujoco.MjData(model)
    if not hide:
        place(model, data, name, X_tk, len(X_tk))
    else:
        for i in range(model.nbody):
            if model.body(i).name.startswith('v'):
                data.mocap_pos[model.body(i).mocapid[0]] = [0, 0, -1e3]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cam_spec['look']; cam.distance = cam_spec['d']
    cam.azimuth = cam_spec['az']; cam.elevation = cam_spec['el']
    renderer.update_scene(data, camera=cam)
    if name == 'murmuration' and BIRD_MODEL['kind'] == 'pigeon' and not hide:
        for i in range(len(X_tk)):
            x, y, th = X_tk[i]
            add_pigeon(renderer.scene, [x, y, ALT], th, WAVE_T['t'], phase=0.9 * i,
                       flock=i // MURMUR_NPRED['n'])
    if name == 'rockets' and not hide:
        add_sea_crests(renderer.scene, WAVE_T['t'])
        for i in range(len(X_tk)):
            x, z, h = X_tk[i]
            rgb = (0.12, 0.30, 0.85) if i < len(X_tk) - 1 else (0.85, 0.12, 0.12)
            add_rocket(renderer.scene, [x, 0.0, z], h, rgb, thrust=1.0, t=WAVE_T['t'])
    if boom is not None and not hide and name == 'dubins_relative':
        add_explosion(renderer.scene, [X_tk[-1, 0], X_tk[-1, 1], 0.06], boom)
    elif boom is not None and not hide and name == 'rockets':
        add_explosion(renderer.scene, [X_tk[-1, 0], 0.0, X_tk[-1, 1]], boom)
    if hist is not None and not hide:
        scn = renderer.scene
        P = _trail_pts(name, hist)  # (K, n, 3)
        for i in range(P.shape[1]):
            col = np.array([0.95, 0.55, 0.15, 1.0]) if i == 0 else np.array([0.2, 0.55, 0.95, 1.0])
            for k in range(P.shape[0] - 1):
                if np.linalg.norm(P[k + 1, i] - P[k, i]) < 1e-6 or scn.ngeom >= scn.maxgeom - 1:
                    continue  # zero-length connectors give NaN orientations and crash the renderer
                g = scn.geoms[scn.ngeom]; scn.ngeom += 1
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.018, P[k, i], P[k + 1, i])
                g.rgba[:] = col; g.emission = 1.0; g.specular = 0.0
    return renderer.render().astype(float)


def render_views(name, X_tk, model, renderer, cam_spec, hist=None, boom=None):
    """Decals are keyed out: magenta pixels show the background render instead."""
    if name == 'dubins_relative':   # keep both vehicles framed as they move (rockets keep a fixed camera so the islands stay in view)
        zoom = 0.45 if boom is not None else 1.0   # zoom in on the blast once the capture happens
        cam_spec = dict(cam_spec, look=[float(X_tk[:, 0].mean()), float(X_tk[:, 1].mean()), 0.0],
                        d=cam_spec['d'] * zoom)
    A = _shot(name, X_tk, model, renderer, cam_spec, hide=False, hist=hist, boom=boom)
    C = _shot(name, X_tk, model, renderer, cam_spec, hide=True)
    R, G, B = A[..., 0], A[..., 1], A[..., 2]
    key = (R - G > 15) & (B - G > 15) & (G < 0.7 * np.minimum(R, B))  # lit/dimmed magenta
    out = np.where(key[..., None], C, A).astype(np.uint8)
    return Image.fromarray(out).resize((CELL_W, CELL_H), Image.LANCZOS)


def car_icon(rgb, size=(40, 18)):
    """Small car thumbnail for legends: body, cabin, windscreen, four wheels."""
    W, H = size
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    body = tuple(int(255 * c) for c in rgb) + (255,)
    d.rounded_rectangle([1, H * 0.42, W - 1, H * 0.78], radius=3, fill=body)
    d.rounded_rectangle([W * 0.28, H * 0.12, W * 0.72, H * 0.46], radius=3, fill=body)
    d.rectangle([W * 0.33, H * 0.17, W * 0.67, H * 0.40], fill=(230, 240, 250, 255))
    for x in (W * 0.22, W * 0.78):
        d.ellipse([x - 3, H * 0.70, x + 3, H * 0.70 + 6], fill=(25, 25, 28, 255))
    return im


def ied_icon(size=(40, 18)):
    """IED symbol: dark mine body inside a red warning ring."""
    W, H = size
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx, cy = W / 2, H / 2
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=(230, 40, 30, 255), width=3)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(30, 30, 32, 255))
    return im


def rocket_icon(rgb, size=(40, 18)):
    """Small rocket thumbnail for legends: pointed nose, white body, fins, flame."""
    W, H = size
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    body = (235, 238, 245, 255)
    d.polygon([(W * 0.20, H * 0.38), (W * 0.70, H * 0.30), (W * 0.82, H * 0.50), (W * 0.70, H * 0.70), (W * 0.20, H * 0.62)], fill=body)
    d.polygon([(W * 0.82, H * 0.50), (W * 0.97, H * 0.38), (W * 0.97, H * 0.62)], fill=tuple(int(255 * c) for c in rgb) + (255,))
    d.polygon([(W * 0.20, H * 0.38), (W * 0.08, H * 0.22), (W * 0.30, H * 0.38)], fill=tuple(int(255 * c) for c in rgb) + (255,))
    d.polygon([(W * 0.20, H * 0.62), (W * 0.08, H * 0.78), (W * 0.30, H * 0.62)], fill=tuple(int(255 * c) for c in rgb) + (255,))
    d.polygon([(W * 0.08, H * 0.42), (W * 0.00, H * 0.50), (W * 0.08, H * 0.58)], fill=(255, 150, 40, 255))
    return im


def draw_role_legend(canvas, x0, y0, name, font):
    """Pursuer and evader keys side by side on the canvas, just above the vehicle row."""
    d = ImageDraw.Draw(canvas)
    rows = {'dubins_relative': [('Pursuer', (0.85, 0.12, 0.12), (0.80, 0.10, 0.10)),
                                ('Evader', (0.12, 0.30, 0.85), (0.10, 0.25, 0.80))],
            'rockets': [('Pursuer', (0.12, 0.30, 0.85), (0.10, 0.25, 0.80)),
                        ('Evader', (0.85, 0.12, 0.12), (0.80, 0.10, 0.10))]}.get(name)
    if rows is None:
        return
    x = x0 + 4
    pieces = []
    for word, thumb, ink in rows:
        pieces.append((x, word, thumb, ink))
        x = int(x + 40 + 6 + d.textlength(word, font=font) + 18)
    if name == 'dubins_relative':
        x = int(x + 46 + d.textlength('IEDs', font=font) + 6)
    d.rectangle([x0, y0, x, y0 + 24], fill=(255, 255, 255))
    icon = rocket_icon if name == 'rockets' else car_icon
    for px, word, thumb, ink in pieces:
        canvas.paste(icon(thumb), (px, y0 + 3), icon(thumb))
        d.text((px + 46, y0 + 6), word, fill=tuple(int(255 * c) for c in ink), font=font)
    if name == 'dubins_relative':   # IED key on the same line, after the evader
        canvas.paste(ied_icon(), (x, y0 + 3), ied_icon())
        d.text((x + 46, y0 + 6), 'IEDs', fill=(160, 20, 20), font=font)


def draw_ied_key(img, name, font):
    """IED key in the top-left corner of the first vehicle image."""
    if name != 'dubins_relative':
        return
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, 92, 26], fill=(255, 255, 255))
    img.paste(ied_icon(), (8, 6), ied_icon())
    d.text((50, 9), 'IEDs', fill=(160, 20, 20), font=font)


_pending = []


def phase_portrait(name, ro, j, t, diff_now):
    """Last column: relative-plane trajectories of both methods, drawn over the danger zone.

    Benchmark is solid, HJ-Gauss dashed; points are colored by elapsed time.
    The inset number is the current mean state-norm gap.
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    ext = 0.5 if name == 'dubins_relative' else STRIP_EXTENT[name]   # zoom on the capture region
    fig, ax = plt.subplots(figsize=(CELL_W / 100, CELL_H / 100), dpi=100)
    ax.add_patch(plt.Circle((0, 0), DANGER_R if name == 'dubins_relative' else 1.5, fill=True,
                            fc=(1, 0.3, 0.3, 0.12), ec='#d62728', lw=1.8, ls='--'))
    cmap = plt.get_cmap('viridis')
    for tag, ls, lw in [('ref', '-', 2.6), ('hj', (0, (3, 2)), 2.6)]:
        X = ro[tag][:j + 1, :, :2]
        for i in range(X.shape[1]):
            ax.plot(X[:, i, 0], X[:, i, 1], ls=ls, lw=lw, color='#1f5fbf' if tag == 'ref' else '#ff7f0e')
            ax.scatter(X[:, i, 0], X[:, i, 1], c=t[:j + 1], cmap=cmap, s=10, vmin=0, vmax=t[-1])
            ax.scatter(X[-1, i, 0], X[-1, i, 1], s=42, edgecolor='k', facecolor='white', zorder=5)
    ax.scatter([0], [0], marker='s', s=60, color='#1f5fbf', zorder=6)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_aspect('equal')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
    if name == 'dubins_relative':   # capture radius stated where the x label would be
        ax.set_xlabel(f'Capture radius {DANGER_R:.2f}', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('Relative $x_1$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative $x_2$', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight('bold')
    ax.text(0.03, 0.97, 'Solid blue: benchmark\nDashed orange: HJ-Gauss', transform=ax.transAxes, va='top',
            fontsize=8, fontweight='bold')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, metadata={'Software': 'matplotlib'})
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB').resize((CELL_W, CELL_H))


def chart(out, t_now, horizon):
    """State-norm gap chart as a CELL_W x CELL_H image, bold and sentence case."""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(CELL_W / 100, CELL_H / 100), dpi=100)
    ax.plot(out['t'], out['diff_mean'], color='#1f5fbf', lw=2.2, label='Mean')
    ax.plot(out['t'], out['diff_max'], color='#c0392b', lw=1.8, ls='--', label='Maximum')
    ax.axvline(t_now, color='k', lw=1.2)
    ax.set_xlim(0, horizon)
    ax.set_xlabel('Elapsed horizon $t$', fontsize=12, fontweight='bold')
    ax.set_ylabel('State-norm gap', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontweight('bold')
    ax.legend(fontsize=9, frameon=False, loc='upper left', prop={'weight': 'bold', 'size': 9})
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, metadata={'Software': 'matplotlib'})
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB').resize((CELL_W, CELL_H))


DECAL_HALF = {'rockets': 1.0, 'dubins_relative': 0.55, 'murmuration': 0.0}
CAMS = {
    'murmuration': dict(az=-135, el=-42, look=[0, 0, 0.3], d=12.5),
    'rockets': dict(az=-90, el=-4, look=[0, 1.5, -2.0], d=10.0),
    'dubins_relative': dict(az=-135, el=-42, look=[0, 0, 0.0], d=2.4),
}
GZ = {'rockets': -3.4, 'dubins_relative': 0.0, 'murmuration': -0.6}
TOP_X = {'rockets': [26, 333, 640], 'dubins_relative': [22, 329, 636], 'murmuration': [26, 333, 640]}
GIF = {'rockets': 'roysoc_rockets.gif', 'dubins_relative': 'roysoc_dubins.gif', 'murmuration': 'roysoc_murmuration.gif'}


def model_assets(name):
    files = {}
    if name == 'dubins_relative':
        for f in ['prius/hybrid_body.obj', 'prius/wheel.obj']:
            files[f] = (ASSETS / f).read_bytes()
    else:
        files = {k: v.read_bytes() for k, v in DECAL_FILES.items()}
    return files


ROLE_TEXT = {'dubins_relative': 'Red: pursuer    Blue: evader', 'rockets': 'Blue: pursuer    Red: evader', 'murmuration': 'Drones: flock members'}
# big bold event title above each composite
EVENT_TITLE = {'rockets': 'Rocket Launch Pursuit-Evasion',
               'dubins_relative': 'Pursuit-Evasion Dubins Vehicle Games',
               'murmuration': 'Cordon Formation'}
TITLE_H = 52
SUB_FRAMES = {'dubins_relative': 8}    # sub-frames per snapshot interval
STEP_MS = {'dubins_relative': 1000}  # milliseconds per snapshot interval (slower pursuit)
STRIP_EXTENT = {'rockets': 3.0, 'dubins_relative': 2.0, 'murmuration': 3.0}
STRIP_LABELS = ['Ground Truth', 'HJ-Gauss', 'Abs. Error']


def snapshot_times(hj_root, name):
    ref_key, _ = FIELDS[name]
    with np.load(Path(hj_root) / (ref_key + '.npz')) as d:
        return np.array(sorted(float(k[2:]) for k in d.files if k.startswith('t_')))


def value_strip(name, hj_root, times, gap=6):
    """Top row rebuilt from the saved archives: sentence case, bold, large type.

    Returns one full-width image per snapshot time, height CELL_H + 2 * gap.
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    ref_key, hj_key = FIELDS[name]
    fr = np.load(Path(hj_root) / (ref_key + '.npz')); fh = np.load(Path(hj_root) / (hj_key + '.npz'))
    ext = STRIP_EXTENT[name]
    ax0, ax1 = fr['axis_0'], fr['axis_1']
    ith = int(np.argmin(np.abs(F.wrap(fr['axis_2']))))  # heading slice nearest zero
    ix = np.where(np.abs(ax0) <= ext + 1e-9)[0]; iy = np.where(np.abs(ax1) <= ext + 1e-9)[0]
    X, Y = np.meshgrid(ax0[ix], ax1[iy], indexing='ij')
    W = 3 * CELL_W + 4 * gap; H = CELL_H + 2 * gap
    out = []
    for t in times:
        key = 't_%g' % t
        a = fr[key][np.ix_(ix, iy, [ith])][..., 0]; b = fh[key][np.ix_(ix, iy, [ith])][..., 0]
        err = np.abs(a - b)
        fig, axs = plt.subplots(1, 3, figsize=(W / 100, H / 100), dpi=100)
        fig.subplots_adjust(left=0.07, right=0.90, bottom=0.27, top=0.86, wspace=0.42)
        vmax = max(1e-6, float(np.max(np.abs(np.r_[a.ravel(), b.ravel()]))))
        emax = max(1e-9, float(err.max()))
        for ax, z, lab in zip(axs, [a, b, err], STRIP_LABELS):
            if lab == 'Abs. Error':
                cf = ax.contourf(X, Y, z, levels=np.linspace(0, emax, 25), cmap='magma', extend='max')
                cax = fig.add_axes([0.915, 0.27, 0.010, 0.59])
                cb = fig.colorbar(cf, cax=cax)
                cb.set_ticks(np.linspace(0, emax, 4))
                cb.set_ticklabels(['%.1e' % v for v in np.linspace(0, emax, 4)])
                cb.ax.tick_params(labelsize=8)
                for tl in cb.ax.get_yticklabels():
                    tl.set_fontweight('bold')
            else:
                ax.contourf(X, Y, z, levels=np.linspace(-vmax, vmax, 25), cmap='RdBu_r', extend='both')
                if z.min() < 0 < z.max():
                    ax.contour(X, Y, z, [0], colors='black', linewidths=1.2)
            ax.set_title(lab, fontsize=14, fontweight='bold')
            ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_aspect('equal')
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=10)
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_fontweight('bold')
        axs[0].set_ylabel('Relative position $x_2$', fontsize=12, fontweight='bold')
        fig.supxlabel('Relative position $x_1$', fontsize=12, fontweight='bold', y=0.13)
        if name == 'dubins_relative':
            for ax in axs:
                ax.add_patch(plt.Circle((0, 0), DANGER_R, fill=False, ec='#d62728', lw=2.0, ls='--'))
            # one legend, inside the ground-truth heat map, lower right
            axs[0].plot([], [], color='black', lw=1.2, label='Capture set')
            axs[0].plot([], [], color='#d62728', lw=2.0, ls='--', label='Danger zone (IEDs)')
            axs[0].plot([], [], color='black', lw=0)
            handles, labels_ = axs[0].get_legend_handles_labels()
            handles += [plt.Line2D([], [], marker='s', ls='', color='#1f5fbf', ms=7),
                        plt.Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=7)]
            labels_ += ['Evader (origin)', 'Pursuer at horizon t']
            fig.legend(handles, labels_, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0),
                       fontsize=9, prop={'weight': 'bold', 'size': 9}, frameon=False)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, metadata={'Software': 'matplotlib'})
        plt.close(fig); buf.seek(0)
        out.append(Image.open(buf).convert('RGB').resize((W, H), Image.LANCZOS))
    return out


def compose(name, hj_root, out_gif, poster, T, dt, anim_dir, sub=None, step_ms=None):
    global fr_rock, fh_rock
    if name == 'rockets':
        fr_rock = F.load_field(Path(hj_root) / (FIELDS[name][0] + '.npz'))
        fh_rock = F.load_field(Path(hj_root) / (FIELDS[name][1] + '.npz'))
    sub = sub or SUB_FRAMES.get(name, 3)
    step_ms = step_ms or STEP_MS.get(name, 400)
    ro = rollouts(name, hj_root, T, dt)
    t = ro['t']
    n = ro['ref'].shape[1]
    norm = lambda X: np.sqrt(np.sum(X ** 2, -1))
    diff = np.abs(norm(ro['ref']) - norm(ro['hj']))
    data = dict(t=t, diff_mean=diff.mean(1), diff_max=diff.max(1))
    xml = scene(name, n, GZ[name])
    model = mujoco.MjModel.from_xml_string(xml, assets=model_assets(name))
    r = mujoco.Renderer(model, CELL_H * SUPER, CELL_W * SUPER, max_geom=20000)
    tv = snapshot_times(hj_root, name)
    top = value_strip(name, hj_root, tv)
    font = ImageFont.truetype(font_manager.findfont('DejaVu Sans:bold'), 13)
    gap = 6
    cap = {'ref': None, 'hj': None}
    if name == 'rockets':   # capture is decided by the value field at the horizon, shown in the last moments
        for k, fld in (('ref', fr_rock), ('hj', fh_rock)):
            vT = fld[0](float(t[-1]), ro[k][-1:, 0, :])   # relative (x1, x2, heading) at the horizon
            if float(np.ravel(vT)[0]) < 0:
                cap[k] = float(t[-1]) - 0.1
    elif name in CAPTURE_R:   # first time the pursuer is inside the capture set
        for k in ('ref', 'hj'):
            if name == 'dubins_relative':   # collision: evader comes within reach of an individual IED
                E = ro[k + '_world'][:, 1, :2]
                ied = np.array(IED_POINTS, float)
                dist = np.min(np.linalg.norm(E[:, None, :] - ied[None, :, :], axis=-1), axis=1)
            else:
                dist = np.hypot(ro[k][:, 0, 0], ro[k][:, 0, 1])
            hit = np.where(dist < CAPTURE_R[name])[0]
            cap[k] = float(t[hit[0]]) if hit.size else None
    boom_font = ImageFont.truetype(font_manager.findfont('DejaVu Sans:bold'), 30)
    FOOT = 40                      # strip under row two: horizon caption and bar
    HEAD = 0                       # no band between the rows: bottom row sits directly under the top row
    ROW2 = CELL_H + 2 * gap + HEAD
    W, H = 3 * CELL_W + 4 * gap, ROW2 + CELL_H + FOOT
    out, durs = [], []
    for k, fr in enumerate(top):
        last = k == len(top) - 1
        steps = 1 if last else sub
        for s in range(steps):
            tau = tv[k] + (tv[min(k + 1, len(tv) - 1)] - tv[k]) * s / steps
            j = int(np.clip(round(tau / dt), 0, len(t) - 1))
            WAVE_T['t'] = float(t[j])
            canv = Image.new('RGB', (W, H), (255, 255, 255))
            canv.paste(fr, (0, 0))
            # horizon axis label and progress bar, outside the vehicle cells
            bd = ImageDraw.Draw(canv)
            bx0, bx1 = gap, W - gap
            bd.text((W / 2, ROW2 + CELL_H + 12), f'Elapsed horizon t = {t[j]:.2f} / {T:g}',
                    fill=(20, 20, 20), font=font, anchor='mm')
            bar_y = ROW2 + CELL_H + 26
            bd.rectangle([bx0, bar_y, bx1, bar_y + 8], fill=(225, 225, 225))
            bd.rectangle([bx0, bar_y, bx0 + (bx1 - bx0) * t[j] / T, bar_y + 8], fill=(200, 60, 40))
            # bottom row
            W_ref = ro['ref_world'] if name in ('dubins_relative', 'rockets', 'murmuration') else ro['ref']
            W_hj = ro['hj_world'] if name in ('dubins_relative', 'rockets', 'murmuration') else ro['hj']
            age = {k: (t[j] - cap[k]) if cap[k] is not None and t[j] >= cap[k] else None for k in ('ref', 'hj')}
            img_ref = render_views(name, W_ref[j], model, r, CAMS[name], W_ref[:j + 1], boom=age['ref'])
            img_hj = render_views(name, W_hj[j], model, r, CAMS[name], W_hj[:j + 1], boom=age['hj'])
            img_ch = phase_portrait(name, ro, j, t, diff_now=float(data['diff_mean'][j]))
            for c, img in enumerate([img_ref, img_hj, img_ch]):
                img = img.copy()
                if c < 2 and age[['ref', 'hj'][c]] is not None:
                    d = ImageDraw.Draw(img)
                    d.text((CELL_W / 2, CELL_H / 2 - 70), 'CAPTURE!', font=boom_font, anchor='mm',
                           fill=(255, 220, 40), stroke_width=4, stroke_fill=(0, 0, 0))
                canv.paste(img, (gap + c * (CELL_W + gap), ROW2))
            draw_role_legend(canv, gap, ROW2 - 28, name, font)
            if ROW2_HEADER['text']:
                header_font = ImageFont.truetype(font_manager.findfont('DejaVu Sans:bold'), 19)
                hd = ImageDraw.Draw(canv)
                hd.rectangle([gap, ROW2 + 2, gap + 8 + hd.textlength(ROW2_HEADER['text'], font=header_font), ROW2 + 30], fill=(255, 255, 255))
                hd.text((gap + 4, ROW2 + 16), ROW2_HEADER['text'], fill=(15, 15, 15), font=header_font, anchor='lm')
            out.append(canv); durs.append(900 if last else step_ms // steps)
    tfont = ImageFont.truetype(font_manager.findfont('DejaVu Sans:bold'), 30)
    titled = []
    if os.environ.get('MATCHED_NO_TITLE'):   # published variant: no event title on the image
        out = [o for o in out]
    for canv in out:
        full = Image.new('RGB', (canv.width, canv.height + TITLE_H), (255, 255, 255))
        d = ImageDraw.Draw(full)
        d.text((canv.width / 2, TITLE_H / 2), EVENT_TITLE[name], fill=(15, 15, 15), font=tfont, anchor='mm')
        full.paste(canv, (0, TITLE_H))
        titled.append(full)
    out = titled if not os.environ.get('MATCHED_NO_TITLE') else out
    pal = [f.convert('P', palette=Image.Palette.ADAPTIVE, colors=256) for f in out]
    pal[0].save(out_gif, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=2)
    out[-1].save(poster)
    return out_gif


ROW2_HEADER = {'text': None}   # header above the bottom row, set per murmuration event
BIRD_MODEL = {'kind': 'quad'}   # 'quad' or 'pigeon', set per murmuration event
MURMUR_EVENT_TITLE = {'cordon': 'Cordon Formation', 'splitting': 'Flock Splitting',
                      'flash': 'Flash Expansion', 'vacuole': 'Vacuole Nucleation',
                      'flocking': 'Flocking Cohesion'}
# predator layout per event, mirrors hjgauss.validation.cases.MURMUR_EVENTS
MURMUR_EVENTS = {'cordon': (7, 1.15, 0.6), 'splitting': (7, 1.9, 0.6), 'flocking': (7, 0.4, 0.6),
                 'vacuole': (6, 1.15, 0.6), 'flash': (1, 0.0, 0.5)}


MURMUR_NPRED = {'n': 7}
MURMUR_CENTRES = {'c': np.zeros((1, 2))}
MURMUR_FLOCKS = {'cordon': 3, 'splitting': 3, 'vacuole': 3, 'flash': 1, 'flocking': 1}
FLOCK_SPACING = 2.4   # distance of each flock's centre from the display origin


def set_murmuration_event(event):
    """Point the renderer at one murmuration event: its value fields, drone starts, and title."""
    npred, ring, rd = MURMUR_EVENTS[event]
    suffix = '' if event == 'cordon' else '_' + event
    FIELDS['murmuration'] = ('murmuration%s_reference_160_400_L6_c0.2' % suffix,
                             'murmuration%s_full_160_400_L6_c0.2' % suffix)
    rows = []
    for k in range(npred):
        a = 2 * np.pi * k / npred
        c = np.array([ring * np.cos(a), ring * np.sin(a)])
        q = c + 0.25 * np.array([np.cos(a + np.pi), np.sin(a + np.pi)])   # inside each disk, facing centre
        rows.append([q[0], q[1], np.arctan2(-q[1], -q[0])])
    if event == 'flash':   # a single predator on the axis would stay still by symmetry; start it off-axis
        rows = [[-0.25, 0.2, 0.5]]
    MURMUR_NPRED['n'] = len(rows)
    nf = MURMUR_FLOCKS[event]
    allrows, centres = [], []
    for f in range(nf):
        rot = 0.45 * f   # each flock starts from a rotated arrangement of the same solved field
        for r in rows:
            c, sn = np.cos(rot), np.sin(rot)
            allrows.append([c * r[0] - sn * r[1], sn * r[0] + c * r[1], r[2] + rot])
        ang = np.pi / 2 + 2 * np.pi * f / nf
        centres.append([FLOCK_SPACING * np.cos(ang) if nf > 1 else 0.0, FLOCK_SPACING * np.sin(ang) if nf > 1 else 0.0])
    STARTS['murmuration'] = np.array(allrows)
    MURMUR_CENTRES['c'] = np.array(centres)
    BIRD_MODEL['kind'] = 'pigeon'   # every murmuration event uses flapping pigeons
    EVENT_TITLE['murmuration'] = MURMUR_EVENT_TITLE[event]
    ROW2_HEADER['text'] = None if event == 'flocking' else ('%d Murmuration Flock%s' % (nf, '' if nf == 1 else 's'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=list(FIELDS), required=True)
    ap.add_argument('--hj-root', type=Path, required=True, help='directory of value-field .npz archives')
    ap.add_argument('--anim-dir', type=Path, required=True, help='directory holding the original 1x3 value GIFs')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--T', type=float, default=None)
    ap.add_argument('--event', choices=list(MURMUR_EVENTS), default='cordon')
    a = ap.parse_args()
    if a.case == 'murmuration':
        set_murmuration_event(a.event)
    T = a.T or {'rockets': 0.5, 'dubins_relative': 1.0, 'murmuration': 2.0}[a.case]
    compose(a.case, a.hj_root, a.out, a.out.with_name(a.out.stem + '_poster.png'), T, T / 200, a.anim_dir)
    print('wrote', a.out)
