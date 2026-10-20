#!/usr/bin/env python3
"""Export the supplied Isaac/Omniverse USDZ into browser preview assets.

The exporter keeps the NuRec tensor conventions intact and converts the USD
render meshes into glTF with transformed normals and the authored PBR values.
It is deliberately a build step: the website never pretends that a generic
browser renderer is Isaac Sim's renderer.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import msgpack
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade


SH_C0 = 0.28209479177387814
DEFAULT_MATERIAL = "__default__"


def to_yup(points: np.ndarray) -> np.ndarray:
    # Isaac/USD uses Z-up; the browser scene uses Y-up.
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def gf_matrix(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)


def transformed_points(points, matrix):
    m = gf_matrix(matrix)
    p = np.concatenate([np.asarray(points, dtype=np.float64), np.ones((len(points), 1))], axis=1)
    return to_yup((p @ m)[:, :3])


def transformed_normals(normals, matrix):
    m = gf_matrix(matrix)[:3, :3]
    # Gf uses row vectors. Normals use the inverse transpose of its linear map.
    c = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    out = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(m).T @ c
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def triangulated_faces(counts, indices):
    faces = []
    cursor = 0
    for face_index, count in enumerate(counts):
        poly = list(indices[cursor : cursor + count])
        cursor += count
        if count < 3:
            continue
        for i in range(1, count - 1):
            # (x,y,z) -> (x,z,-y) is a rotation, so winding stays unchanged.
            faces.append((face_index, poly[0], poly[i], poly[i + 1]))
    return faces


def material_shader(material):
    if not material:
        return None
    for prim in Usd.PrimRange(material.GetPrim()):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() or shader.GetInputs():
                return shader
    return None


def input_value(shader, names, default=None):
    if not shader:
        return default
    for name in names:
        inp = shader.GetInput(name)
        if not inp:
            continue
        value = inp.Get()
        if value is not None:
            return value
    return default


def vec3(value, default):
    if value is None:
        return np.asarray(default, dtype=np.float32)
    try:
        return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    except (TypeError, IndexError):
        return np.asarray(default, dtype=np.float32)


def fallback_material(path):
    path = str(path)
    if "silver_fork" in path:
        return np.array([0.8, 0.8, 0.8], dtype=np.float32), 1.0, 0.08
    if "Robotiq" in path:
        return np.array([0.16, 0.16, 0.17], dtype=np.float32), 0.35, 0.48
    if "blue_cutlery" in path:
        return np.array([0.036, 0.068, 0.324], dtype=np.float32), 0.0, 0.5
    return np.array([0.72, 0.74, 0.78], dtype=np.float32), 0.0, 0.55


class MaterialTable:
    def __init__(self):
        self.records = []
        self.by_key = {}

    def get(self, material, fallback_path):
        key = str(material.GetPath()) if material else f"{DEFAULT_MATERIAL}:{fallback_path}"
        if key in self.by_key:
            return self.by_key[key]
        base, metallic, roughness = fallback_material(fallback_path)
        shader = material_shader(material)
        authored = input_value(shader, ("diffuseColor", "diffuse_color_constant", "diffuse_color", "baseColor", "albedo"))
        base = np.clip(vec3(authored, base), 0.0, 1.0)
        metallic = float(input_value(shader, ("metallic", "metallic_constant"), metallic))
        roughness = float(input_value(shader, ("roughness", "reflection_roughness_constant"), roughness))
        emission_enabled = input_value(shader, ("enable_emission",), False)
        emission = vec3(input_value(shader, ("emissiveColor", "emissive_color")), [0.0, 0.0, 0.0]) if emission_enabled else np.zeros(3, np.float32)
        record = {
            "name": key,
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(x) for x in base] + [1.0],
                "metallicFactor": float(np.clip(metallic, 0.0, 1.0)),
                "roughnessFactor": float(np.clip(roughness, 0.0, 1.0)),
            },
            "doubleSided": False,
        }
        if np.max(emission) > 0:
            record["emissiveFactor"] = [float(x) for x in np.clip(emission, 0.0, 1.0)]
        index = len(self.records)
        self.records.append(record)
        self.by_key[key] = index
        return index


def mesh_materials(prim, face_count, table):
    base = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    out = [table.get(base, prim.GetPath())] * face_count
    for subset in prim.GetChildren():
        if not subset.IsA(UsdGeom.Subset):
            continue
        material = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()[0]
        if not material:
            continue
        indices = UsdGeom.Subset(subset).GetIndicesAttr().Get() or []
        material_index = table.get(material, subset.GetPath())
        for face_index in indices:
            if 0 <= int(face_index) < face_count:
                out[int(face_index)] = material_index
    return out


def visible_render_mesh(prim):
    path = str(prim.GetPath())
    if "/collision/" in path or "TableCollider" in path:
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return False
    if str(imageable.ComputePurpose()) not in ("default", "render"):
        return False
    return True


def export_meshes(stage, output):
    cache = UsdGeom.XformCache()
    table = MaterialTable()
    groups = defaultdict(lambda: {"positions": [], "normals": [], "indices": []})
    mesh_count = 0
    triangle_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not visible_render_mesh(prim):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        raw_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not raw_indices:
            continue
        triangles = triangulated_faces(counts, raw_indices)
        if not triangles:
            continue
        world_points = transformed_points(points, cache.GetLocalToWorldTransform(prim)).astype(np.float32)
        raw_normals = mesh.GetNormalsAttr().Get() or []
        interpolation = str(mesh.GetNormalsInterpolation() or "")
        if raw_normals and interpolation == "faceVarying" and len(raw_normals) >= sum(int(x) for x in counts):
            world_normals = transformed_normals(raw_normals, cache.GetLocalToWorldTransform(prim)).astype(np.float32)
        elif raw_normals and len(raw_normals) == len(points):
            world_normals = transformed_normals(raw_normals, cache.GetLocalToWorldTransform(prim)).astype(np.float32)
        else:
            world_normals = None
        face_materials = mesh_materials(prim, len(counts), table)
        group_offsets = defaultdict(int)
        face_corner_start = 0
        face_corner_lookup = {}
        for face_index, count in enumerate(counts):
            face_corner_lookup[face_index] = face_corner_start
            face_corner_start += int(count)
        for face_index, a, b, c in triangles:
            material_index = face_materials[face_index]
            group = groups[material_index]
            base = len(group["positions"])
            verts = [a, b, c]
            p = world_points[verts]
            if world_normals is None:
                n = np.cross(p[1] - p[0], p[2] - p[0])
                n /= max(float(np.linalg.norm(n)), 1e-12)
                normals = np.repeat(n[None, :], 3, axis=0)
            elif interpolation == "faceVarying":
                start = face_corner_lookup[face_index]
                normals = world_normals[[start + next(i for i, value in enumerate(raw_indices[start : start + int(counts[face_index])]) if value == vertex) for vertex in verts]]
            else:
                normals = world_normals[verts]
            group["positions"].extend(p.tolist())
            group["normals"].extend(normals.tolist())
            group["indices"].extend([base, base + 1, base + 2])
            triangle_count += 1
        mesh_count += 1
    primitives = []
    for material_index, group in groups.items():
        if not group["positions"]:
            continue
        primitives.append({
            "positions": np.asarray(group["positions"], dtype=np.float32),
            "normals": np.asarray(group["normals"], dtype=np.float32),
            "indices": np.asarray(group["indices"], dtype=np.uint32),
            "material": int(material_index),
        })
    write_glb(primitives, table.records, output)
    print(f"render meshes: {mesh_count}, triangles: {triangle_count}, primitives: {len(primitives)}")


def pad(data, fill=b"\x00"):
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(primitives, materials, path):
    blob = bytearray()
    views = []
    accessors = []

    def add_view(data, target):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(view, component_type, count, type_name, minimum=None, maximum=None):
        accessor = {"bufferView": view, "componentType": component_type, "count": count, "type": type_name}
        if minimum is not None:
            accessor["min"] = [float(x) for x in minimum]
            accessor["max"] = [float(x) for x in maximum]
        accessors.append(accessor)
        return len(accessors) - 1

    gltf_primitives = []
    for primitive in primitives:
        pos = primitive["positions"]
        norm = primitive["normals"]
        idx = primitive["indices"]
        vp = add_view(pos.astype("<f4").tobytes(), 34962)
        vn = add_view(norm.astype("<f4").tobytes(), 34962)
        vi = add_view(idx.astype("<u4").tobytes(), 34963)
        ap = add_accessor(vp, 5126, len(pos), "VEC3", pos.min(axis=0), pos.max(axis=0))
        an = add_accessor(vn, 5126, len(norm), "VEC3")
        ai = add_accessor(vi, 5125, len(idx), "SCALAR")
        gltf_primitives.append({"attributes": {"POSITION": ap, "NORMAL": an}, "indices": ai, "material": primitive["material"], "mode": 4})
    document = {
        "asset": {"version": "2.0", "generator": "WorldMirror USDZ exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "IsaacSim USD render meshes"}],
        "meshes": [{"name": "WorldMirror IsaacSim render meshes", "primitives": gltf_primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    # glTF requires JSON chunks padded with spaces (0x20), not NUL.
    json_bytes = pad(json.dumps(document, separators=(",", ":")).encode("utf-8"), fill=b" ")
    bin_bytes = pad(bytes(blob))
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        handle.write(json_bytes)
        handle.write(struct.pack("<I4s", len(bin_bytes), b"BIN\x00"))
        handle.write(bin_bytes)


def export_gaussians(stage, usdz, output, threshold):
    volume = stage.GetPrimAtPath("/World/Room/NuRec/gauss/gauss")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(volume)
    with zipfile.ZipFile(usdz) as archive:
        packed = msgpack.unpackb(gzip.decompress(archive.read("nurec.nurec")), raw=False)
    state = packed["nre_data"]["state_dict"]

    def load(name):
        shape = state[name + ".shape"]
        return np.frombuffer(state[name], dtype=np.float16).astype(np.float32).reshape(shape)

    positions = load(".gaussians_nodes.gaussians.positions")
    rotations = load(".gaussians_nodes.gaussians.rotations")
    scales = load(".gaussians_nodes.gaussians.scales")
    densities = load(".gaussians_nodes.gaussians.densities").reshape(-1)
    albedo = load(".gaussians_nodes.gaussians.features_albedo")
    opacity = 1.0 / (1.0 + np.exp(-np.clip(densities, -20, 20)))
    keep = opacity > threshold
    positions, rotations, scales, densities, albedo = (x[keep] for x in (positions, rotations, scales, densities, albedo))

    m = gf_matrix(matrix)
    p4 = np.concatenate([positions.astype(np.float64), np.ones((len(positions), 1))], axis=1)
    world = (p4 @ m)[:, :3]
    positions = to_yup(world).astype(np.float32)

    # Both NuRec and the INRIA PLY format store wxyz. The viewer only
    # converts to Three.js xyzw when constructing the covariance matrix.
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    # This scene's volume transform followed by Z-up -> Y-up is uniform
    # scale plus translation. No further quaternion rotation is needed.
    c = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    linear = m[:3,:3] @ c
    factor = np.linalg.norm(linear[0])
    if not np.allclose(linear, np.eye(3) * factor, atol=1e-7):
        raise ValueError("This scene needs an additional rotation in its Gaussian export")

    # NuRec stores pre-activation log scales. The viewer expects log scales;
    # apply the volume's uniform scale before writing them.
    scale_factor = float(np.linalg.norm(m[:3, 0]))
    scales = (scales + math.log(scale_factor)).astype(np.float32)

    count = len(densities)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {count}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
              "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
              "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
              "end_header\n").encode()
    record = np.empty(count, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("dc0", "<f4"), ("dc1", "<f4"), ("dc2", "<f4"), ("op", "<f4"),
                                    ("s0", "<f4"), ("s1", "<f4"), ("s2", "<f4"),
                                    ("r0", "<f4"), ("r1", "<f4"), ("r2", "<f4"), ("r3", "<f4")])
    record["x"], record["y"], record["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    record["dc0"], record["dc1"], record["dc2"] = albedo[:, 0], albedo[:, 1], albedo[:, 2]
    record["op"] = densities
    record["s0"], record["s1"], record["s2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    record["r0"], record["r1"], record["r2"], record["r3"] = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    with open(output, "wb") as handle:
        handle.write(header)
        record.tofile(handle)
    print(f"gaussians: {count} (opacity > {threshold}), bounds: {positions.min(0)} .. {positions.max(0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usdz", type=Path, default=Path("scene_selfcontained.usdz"))
    parser.add_argument("--ply", type=Path, default=Path("scene/room-native.ply"))
    parser.add_argument("--glb", type=Path, default=Path("scene/props-native.glb"))
    parser.add_argument("--opacity-threshold", type=float, default=0.011)
    args = parser.parse_args()
    args.ply.parent.mkdir(parents=True, exist_ok=True)
    args.glb.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(args.usdz))
    if not stage:
        raise SystemExit(f"Could not open {args.usdz}")
    export_gaussians(stage, args.usdz, args.ply, args.opacity_threshold)
    export_meshes(stage, args.glb)


if __name__ == "__main__":
    main()
