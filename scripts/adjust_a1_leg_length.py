#!/usr/bin/env python3
"""Set A1 URDF leg segment lengths (joint offsets + leg collision boxes) for chosen legs only.

Along-leg box size uses the URDF convention (first scalar); lateral thickness (other two scalars)
is untouched. Meshes/inertia are unchanged."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

_LEG = frozenset({"FR", "FL", "RR", "RL"})
_DEFAULT_URDF = Path(__file__).resolve().parents[1] / "robots/a1/a1_description/urdf/a1.urdf"


def _set_origin_z(elem: ET.Element, z: float) -> None:
    o = elem.find("origin")
    x, y, _ = map(float, o.attrib["xyz"].split())
    o.set("xyz", f"{x} {y} {z}")


def _patch_leg_collision(link: ET.Element, length: float) -> None:
    for col in link.findall("collision"):
        box = col.find("geometry/box")
        if box is None:
            continue
        _, sy, sz = box.attrib["size"].split()
        box.set("size", f"{length} {sy} {sz}")
        _set_origin_z(col, -0.5 * length)


def _joint_prefix(joint_name: str) -> str | None:
    p, *rest = joint_name.split("_", 2)
    if p not in _LEG:
        return None
    if joint_name.endswith("_lower_joint") or joint_name.endswith("_toe_fixed"):
        return p
    return None


def apply_lengths(
    robot: ET.Element,
    thigh: float,
    calf: float,
    legs: frozenset[str],
    part: str,
) -> None:
    do_thigh = part in ("thigh", "both")
    do_calf = part in ("calf", "both")

    for j in robot.findall("joint"):
        name = j.attrib.get("name", "")
        pfx = _joint_prefix(name)
        if pfx is None or pfx not in legs:
            continue
        if name.endswith("_lower_joint") and do_thigh:
            _set_origin_z(j, -thigh)
        elif name.endswith("_toe_fixed") and do_calf:
            _set_origin_z(j, -calf)

    for link in robot.findall("link"):
        name = link.attrib.get("name", "")
        if name.split("_", 1)[0] not in legs:
            continue
        if do_thigh and name.endswith("_upper"):
            _patch_leg_collision(link, thigh)
        elif do_calf and name.endswith("_lower"):
            _patch_leg_collision(link, calf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=_DEFAULT_URDF)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path.cwd() / "a1.urdf",
        help="Output URDF (default: ./a1.urdf under current working directory).",
    )
    ap.add_argument(
        "--legs",
        nargs="+",
        default=None,
        choices=sorted(_LEG),
        metavar="LEG",
        help="One or more of FR FL RR RL. Default: all legs.",
    )
    ap.add_argument(
        "--part",
        choices=("thigh", "calf", "both"),
        default="both",
        help="'calf': only shin (lower) length; 'thigh': only thigh; 'both'.",
    )
    ap.add_argument("--thigh", type=float, default=0.2, help="Thigh segment length along -Z (m).")
    ap.add_argument("--calf", type=float, default=0.2, help="Shin segment length along -Z (m).")
    args = ap.parse_args()

    if args.part in ("thigh", "both") and args.thigh <= 0:
        raise SystemExit("--thigh must be positive")
    if args.part in ("calf", "both") and args.calf <= 0:
        raise SystemExit("--calf must be positive")

    legs = frozenset(args.legs if args.legs is not None else _LEG)

    tree = ET.parse(args.urdf)
    root = tree.getroot()
    apply_lengths(root, args.thigh, args.calf, legs, args.part)
    ET.indent(tree, space="  ")
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    main()
