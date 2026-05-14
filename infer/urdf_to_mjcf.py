"""Generate a MuJoCo MJCF string from the bundled Unitree A1 URDF (fixed + revolute only)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _rpy_to_R(r: float, p: float, y: float) -> list[list[float]]:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _R_to_quat_wxyz(R: list[list[float]]) -> tuple[float, float, float, float]:
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        w, x, y, z = (R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        w, x, y, z = (R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        w, x, y, z = (R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


def _parse_xyz(txt: str | None, default: str = "0 0 0") -> tuple[float, float, float]:
    if not txt:
        txt = default
    p = [float(x) for x in txt.split()]
    if len(p) != 3:
        raise ValueError(txt)
    return p[0], p[1], p[2]


def _joint_origin(joint_el: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = joint_el.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return _parse_xyz(origin.get("xyz")), _parse_xyz(origin.get("rpy"))


def _rpy_to_quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    return _R_to_quat_wxyz(_rpy_to_R(rpy[0], rpy[1], rpy[2]))


def _parse_inertial(link_el: ET.Element) -> dict[str, Any]:
    inertial = link_el.find("inertial")
    if inertial is None:
        return {"mass": 1e-3, "pos": (0.0, 0.0, 0.0), "ixx": 1e-6, "ixy": 0.0, "ixz": 0.0, "iyy": 1e-6, "iyz": 0.0, "izz": 1e-6}
    mass = float(inertial.find("mass").get("value"))
    origin = inertial.find("origin")
    pos = _parse_xyz(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
    inertia = inertial.find("inertia")
    return {
        "mass": mass,
        "pos": pos,
        "ixx": float(inertia.get("ixx")),
        "ixy": float(inertia.get("ixy", "0")),
        "ixz": float(inertia.get("ixz", "0")),
        "iyy": float(inertia.get("iyy")),
        "iyz": float(inertia.get("iyz", "0")),
        "izz": float(inertia.get("izz")),
    }


def _geom_tags(link_el: ET.Element, link_name: str, vis_cls: str, col_cls: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tag, cls, pfx in (
        ("visual", vis_cls, "v"),
        ("collision", col_cls, "c"),
    ):
        for idx, el in enumerate(link_el.findall(tag)):
            g = el.find("geometry")
            if g is None:
                continue
            origin = el.find("origin")
            pos = _parse_xyz(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
            rpy = _parse_xyz(origin.get("rpy")) if origin is not None else (0.0, 0.0, 0.0)
            mesh = g.find("mesh")
            box = g.find("box")
            cyl = g.find("cylinder")
            sph = g.find("sphere")
            name = f"{link_name}_{pfx}{idx}"
            if mesh is not None:
                out.append({"cls": cls, "kind": "mesh", "name": name, "pos": pos, "rpy": rpy, "file": mesh.get("filename", "")})
            elif box is not None:
                sx, sy, sz = _parse_xyz(box.get("size"))
                out.append({"cls": cls, "kind": "box", "name": name, "pos": pos, "rpy": rpy, "half": (sx * 0.5, sy * 0.5, sz * 0.5)})
            elif cyl is not None:
                out.append(
                    {
                        "cls": cls,
                        "kind": "cyl",
                        "name": name,
                        "pos": pos,
                        "rpy": rpy,
                        "radius": float(cyl.get("radius")),
                        "half_h": float(cyl.get("length")) * 0.5,
                    }
                )
            elif sph is not None:
                out.append({"cls": cls, "kind": "sph", "name": name, "pos": pos, "rpy": rpy, "r": float(sph.get("radius"))})
    return out


def _emit_geoms(lines: list[str], geoms: list[dict[str, Any]], ind: str) -> None:
    for g in geoms:
        wxyz = _rpy_to_quat(g["rpy"])
        quat = f"{wxyz[0]} {wxyz[1]} {wxyz[2]} {wxyz[3]}"
        pos = f'{g["pos"][0]} {g["pos"][1]} {g["pos"][2]}'
        cls = g["cls"]
        if g["kind"] == "mesh":
            fn = Path(g["file"]).name
            lines.append(f'{ind}<geom class="{cls}" mesh="{fn}" pos="{pos}" quat="{quat}"/>')
        elif g["kind"] == "box":
            hx, hy, hz = g["half"]
            lines.append(f'{ind}<geom class="{cls}" type="box" size="{hx} {hy} {hz}" pos="{pos}" quat="{quat}"/>')
        elif g["kind"] == "cyl":
            lines.append(
                f'{ind}<geom class="{cls}" type="cylinder" size="{g["radius"]} {g["half_h"]}" pos="{pos}" quat="{quat}"/>'
            )
        elif g["kind"] == "sph":
            lines.append(f'{ind}<geom class="{cls}" type="sphere" size="{g["r"]}" pos="{pos}" quat="{quat}"/>')


def _emit_inertial(lines: list[str], inertia: dict[str, Any], ind: str) -> None:
    c = inertia["pos"]
    fi = f'{inertia["ixx"]} {inertia["iyy"]} {inertia["izz"]} {inertia["ixy"]} {inertia["ixz"]} {inertia["iyz"]}'
    lines.append(f'{ind}<inertial pos="{c[0]} {c[1]} {c[2]}" mass="{inertia["mass"]}" fullinertia="{fi}"/>')


def generate_a1_mjcf_from_urdf(
    urdf_path: Path,
    *,
    meshdir: Path | None = None,
    timestep: float = 0.001,
    trunk_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.32),
) -> str:
    urdf_path = urdf_path.resolve()
    if meshdir is None:
        meshdir = urdf_path.parent.parent / "meshes"
    meshdir = meshdir.resolve()

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links: dict[str, ET.Element] = {ln.get("name"): ln for ln in root.findall("link")}
    joints: list[ET.Element] = list(root.findall("joint"))
    children_from_parent: dict[str, list[ET.Element]] = {}
    for j in joints:
        p = j.find("parent").get("link")
        children_from_parent.setdefault(p, []).append(j)

    mesh_names: set[str] = set()
    for ln in links.values():
        for tag in ("visual", "collision"):
            for el in ln.findall(tag):
                g = el.find("geometry")
                if g is None:
                    continue
                m = g.find("mesh")
                if m is not None:
                    mesh_names.add(Path(m.get("filename", "")).name)

    vis_cls, col_cls = "vis", "coll"
    lines: list[str] = []
    lines.append('<mujoco model="a1_genloco">')
    lines.append(f'  <compiler angle="radian" autolimits="true" meshdir="{meshdir.as_posix()}/" eulerseq="xyz"/>')
    lines.append(f'  <option timestep="{timestep}" gravity="0 0 -10" integrator="implicitfast"/>')
    lines.append('  <visual><map force="0.01"/></visual>')
    lines.append("  <default>")
    lines.append('    <default class="vis"><geom group="1" type="mesh" contype="0" conaffinity="0"/></default>')
    lines.append('    <default class="coll"><geom group="0"/></default>')
    lines.append("  </default>")
    lines.append("  <asset>")
    for name in sorted(mesh_names):
        lines.append(f'    <mesh name="{name}" file="{name}"/>')
    # Simple scene visual (checker ground + gradient sky dome)
    lines.append(
        '    <texture type="skybox" builtin="gradient" rgb1="0.42 0.58 0.78" rgb2="0.10 0.12 0.16"'
        ' width="256" height="512"/>'
    )
    lines.append(
        '    <texture name="tex_checker" type="2d" builtin="checker"'
        ' rgb1="0.40 0.62 0.94" rgb2="0.16 0.34 0.72" width="512" height="512" mark="edge" markrgb="0.72 0.84 1.0"/>'
    )
    lines.append(
        '    <material name="mat_floor" texture="tex_checker" texuniform="true" texrepeat="20 20"'
        ' reflectance="0.12" shininess="0.15"/>'
    )
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append(
        '    <geom name="floor" type="plane" size="40 40 0.05" material="mat_floor" group="0"'
        ' condim="3" friction="0.9 0.05 0.002" rgba="1 1 1 1"/>'
    )
    lines.append(
        '    <light name="sun" directional="true" pos="8 8 14" dir="-0.38 -0.38 -1"'
        ' diffuse="0.92 0.90 0.86" castshadow="true"/>'
    )
    lines.append(
        '    <light name="fill" directional="true" pos="-12 10 18" dir="0.32 -0.28 -1"'
        ' diffuse="0.35 0.38 0.45"/>'
    )

    def subtree(link_name: str, indent: str) -> None:
        link_el = links[link_name]
        inertia = _parse_inertial(link_el)
        _emit_inertial(lines, inertia, indent)
        _emit_geoms(lines, _geom_tags(link_el, link_name, vis_cls, col_cls), indent)
        for j in sorted(children_from_parent.get(link_name, []), key=lambda x: x.get("name") or ""):
            child = j.find("child").get("link")
            xyz, rpy = _joint_origin(j)
            wq = _rpy_to_quat(rpy)
            pos = f"{xyz[0]} {xyz[1]} {xyz[2]}"
            quat = f"{wq[0]} {wq[1]} {wq[2]} {wq[3]}"
            jt = j.get("type")
            lines.append(f'{indent}<body name="{child}" pos="{pos}" quat="{quat}">')
            if jt == "fixed":
                subtree(child, indent + "  ")
                lines.append(f"{indent}</body>")
                continue
            if jt != "revolute":
                raise NotImplementedError(f"joint type {jt}")
            axis = _parse_xyz(j.find("axis").get("xyz"))
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            damp_el = j.find("dynamics")
            damping = float(damp_el.get("damping", "0")) if damp_el is not None else 0.0
            jname = j.get("name")
            lines.append(
                f'{indent}  <joint name="{jname}" type="hinge" axis="{axis[0]} {axis[1]} {axis[2]}" '
                f'range="{lo} {hi}" limited="true" damping="{damping}"/>'
            )
            subtree(child, indent + "  ")
            lines.append(f"{indent}</body>")

    px, py, pz = trunk_init_pos
    lines.append(f'  <body name="trunk" pos="{px} {py} {pz}">')
    lines.append("    <freejoint/>")
    subtree("trunk", "    ")
    lines.append("  </body>")
    lines.append("  </worldbody>")

    lines.append("  <actuator>")
    for j in joints:
        if j.get("type") != "revolute":
            continue
        jname = j.get("name")
        eff = float(j.find("limit").get("effort", "33.5"))
        lines.append(f'    <motor name="torque_{jname}" joint="{jname}" gear="1" ctrllimited="true" ctrlrange="{-eff} {eff}"/>')
    lines.append("  </actuator>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


def write_mjcf_cache(urdf_path: Path, out_xml: Path) -> None:
    xml = generate_a1_mjcf_from_urdf(urdf_path)
    out_xml.write_text(xml, encoding="utf-8")
