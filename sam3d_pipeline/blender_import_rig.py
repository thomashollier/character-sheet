"""Blender Python script: Import SAM 3D Body mesh and build a bound skeleton.

Run from command line:
    blender --background --python blender_import_rig.py

Or from within Blender's scripting tab.

Reads exports/character_full_data.json and creates:
  - Mesh object with correct topology
  - Armature with 127-joint skeleton hierarchy
  - Vertex groups with skin weights
  - Exports as FBX and glTF
"""

import bpy
import bmesh
import json
import os
import sys
import math
from mathutils import Vector, Matrix, Quaternion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "exports", "character_full_data.json")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def get_collection():
    """Get or create the active collection."""
    if bpy.context.collection:
        return bpy.context.collection
    if bpy.context.scene.collection:
        return bpy.context.scene.collection
    col = bpy.data.collections.new("SAM3DBody")
    bpy.context.scene.collection.children.link(col)
    return col


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)


def create_mesh(data):
    """Create mesh from vertices and faces."""
    verts = data["pred_vertices"]
    faces = data["mesh_faces_data"]

    mesh = bpy.data.meshes.new("CharacterMesh")
    obj = bpy.data.objects.new("Character", mesh)
    get_collection().objects.link(obj)

    # Build mesh
    bm = bmesh.new()
    bm_verts = []
    for v in verts:
        # SAM 3D Body: X=right-in-image, Y=down, Z=into-screen
        # Blender: X=right, Y=forward, Z=up
        # Negate X to go from image-right to character-right (mirror)
        bm_verts.append(bm.verts.new((-v[0], -v[2], -v[1])))

    bm.verts.ensure_lookup_table()

    n_bad = 0
    for f in faces:
        try:
            face_verts = [bm_verts[f[0]], bm_verts[f[1]], bm_verts[f[2]]]
            bm.faces.new(face_verts)
        except (IndexError, ValueError):
            n_bad += 1

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    print(f"  Mesh: {len(verts)} verts, {len(faces) - n_bad} faces ({n_bad} skipped)")
    return obj


def create_armature(data):
    """Create armature from joint hierarchy."""
    joint_coords = data["pred_joint_coords"]
    joint_parents = data["joint_parents"]
    n_joints = len(joint_coords)

    # Create armature
    arm_data = bpy.data.armatures.new("CharacterRig")
    arm_obj = bpy.data.objects.new("CharacterArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    # Enter edit mode to add bones
    bpy.ops.object.mode_set(mode='EDIT')

    bones = []
    bone_names = []

    for i in range(n_joints):
        name = f"joint_{i:03d}"
        bone_names.append(name)

        bone = arm_data.edit_bones.new(name)
        pos = joint_coords[i]
        # Convert coordinates: SAM (Y-down) → Blender (Z-up), negate X
        head = Vector((-pos[0], -pos[2], -pos[1]))
        # Bone tail: offset slightly along Y (Blender needs non-zero bone length)
        bone.head = head
        bone.tail = head + Vector((0, 0, 0.01))
        bones.append(bone)

    # Set parent relationships
    for i in range(n_joints):
        parent_idx = joint_parents[i]
        if parent_idx >= 0 and parent_idx < n_joints:
            bones[i].parent = bones[parent_idx]
            # Point bone tail toward child for nicer visualization
            parent_bone = bones[parent_idx]
            child_head = bones[i].head
            direction = child_head - parent_bone.head
            if direction.length > 0.005:
                parent_bone.tail = child_head

    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Armature: {n_joints} joints")
    return arm_obj, bone_names


def apply_skin_weights(mesh_obj, arm_obj, bone_names, data):
    """Apply skin weights from MHR skinning data."""
    n_verts = len(data["pred_vertices"])
    n_joints = len(bone_names)

    weights_flat = data.get("skin_weights_flat", [])
    indices_flat = data.get("skin_indices_flat", [])
    vert_indices_flat = data.get("vert_indices_flat", [])

    if not weights_flat or not indices_flat or not vert_indices_flat:
        print("  WARNING: No skinning data, skipping weight assignment")
        return

    # Create vertex groups for each joint
    for name in bone_names:
        mesh_obj.vertex_groups.new(name=name)

    # Build per-vertex weight map
    # The flattened arrays: vert_indices[k] = vertex, indices[k] = joint, weights[k] = weight
    n_entries = len(weights_flat)
    print(f"  Skinning: {n_entries} weight entries across {n_verts} verts, {n_joints} joints")

    assigned = 0
    for k in range(n_entries):
        vi = int(vert_indices_flat[k])
        ji = int(indices_flat[k])
        w = float(weights_flat[k])

        if vi < n_verts and ji < n_joints and w > 1e-6:
            vg = mesh_obj.vertex_groups[bone_names[ji]]
            vg.add([vi], w, 'ADD')
            assigned += 1

    print(f"  Assigned {assigned} weight entries")

    # Parent mesh to armature with armature modifier
    mesh_obj.parent = arm_obj
    modifier = mesh_obj.modifiers.new("Armature", 'ARMATURE')
    modifier.object = arm_obj


def create_camera(data, mesh_obj=None):
    """Create a camera matching SAM 3D Body's projection."""
    focal_length = data.get("focal_length", 1462.5)
    cam_t = data.get("pred_cam_t", [0, 0, 3])
    img_shape = data.get("image_shape", [1024, 1024, 3])
    img_h, img_w = img_shape[0], img_shape[1]

    # SAM 3D Body uses a weak perspective camera:
    #   focal_length is in pixels, image is img_w x img_h
    # Convert to Blender: sensor_width (mm) and focal_length (mm)
    # Blender: focal_mm / sensor_mm = focal_px / img_width_px
    sensor_width = 36.0  # standard 35mm sensor
    focal_mm = focal_length * sensor_width / img_w

    cam_data = bpy.data.cameras.new("SAM3DCamera")
    cam_data.lens = focal_mm
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_width * img_h / img_w
    cam_data.sensor_fit = 'HORIZONTAL'

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    get_collection().objects.link(cam_obj)

    # SAM 3D Body camera model:
    #   Camera at origin, looking down +Z, Y-down
    #   pred_cam_t = body position in camera space
    #   Camera position in body space = -pred_cam_t
    #
    # Axis mapping (same as mesh vertices):
    #   Blender X = SAM X    (right)
    #   Blender Y = -SAM Z   (SAM forward → Blender back-to-front)
    #   Blender Z = -SAM Y   (SAM down → Blender up)
    cam_pos = (
        cam_t[0],            # negate (body→camera) then negate X (mirror) → back to cam_t[0]
        -(-cam_t[2]),        # negate then flip Z
        -(-cam_t[1]),        # negate then flip Y
    )
    cam_obj.location = cam_pos

    # Camera orientation: SAM looks down +Z → Blender -Y
    # Blender camera looks down its local -Z by default.
    # Rotation of 90° around X maps local -Z to world -Y (viewing dir)
    # and local +Y to world +Z (up).
    # Full SAM → Blender camera coordinate conversion:
    #   -90° X: SAM +Z (forward) → Blender -Y (camera looks at character)
    #   180° Y: flips up from -Z to +Z (right-side up) and right from
    #           +X to -X (cancels the X-negation applied to the mesh)
    cam_obj.rotation_euler = (math.radians(-90), math.radians(180), 0)

    # Set as active camera
    bpy.context.scene.camera = cam_obj

    # Set render resolution to match input image
    bpy.context.scene.render.resolution_x = img_w
    bpy.context.scene.render.resolution_y = img_h

    # Set background image (if the source image is available)
    # BASE_DIR is sam3d_pipeline/, parent is multiangle/
    source_img = os.path.join(
        os.path.dirname(BASE_DIR),
        "pose_full_output",
        data.get("filename", "")
    )
    if os.path.exists(source_img):
        img = bpy.data.images.load(source_img)
        cam_data.show_background_images = True
        bg = cam_data.background_images.new()
        bg.image = img
        bg.alpha = 0.5
        bg.display_depth = 'BACK'
        print(f"  Background image: {data.get('filename', '?')}")

    print(f"  Focal length: {focal_mm:.1f}mm (sensor {sensor_width}mm)")
    print(f"  Camera position: {cam_obj.location[:]}")
    print(f"  Render: {img_w}x{img_h}")


def export_fbx(filepath):
    """Export scene as FBX."""
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
    )
    size = os.path.getsize(filepath)
    print(f"  FBX: {filepath} ({size / 1024:.0f} KB)")


def export_gltf(filepath):
    """Export scene as glTF."""
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=False,
        export_skins=True,
        export_animations=False,
    )
    size = os.path.getsize(filepath)
    print(f"  glTF: {filepath} ({size / 1024:.0f} KB)")


def main():
    print("\n=== SAM 3D Body → Blender Rigged Mesh ===")
    print(f"Data: {DATA_PATH}")

    data = load_data()
    print(f"Loaded: {len(data['pred_vertices'])} verts, "
          f"{data.get('mesh_faces', '?')} faces, "
          f"{len(data.get('joint_parents', []))} joints")

    clear_scene()

    # Build mesh
    print("\nCreating mesh...")
    mesh_obj = create_mesh(data)

    # Build armature
    print("\nCreating armature...")
    arm_obj, bone_names = create_armature(data)

    # Apply skinning
    print("\nApplying skin weights...")
    apply_skin_weights(mesh_obj, arm_obj, bone_names, data)

    # Create camera matching SAM 3D Body's projection
    print("\nCreating camera...")
    create_camera(data, mesh_obj)

    # Export
    print("\nExporting...")
    export_fbx(os.path.join(EXPORT_DIR, "character_rigged.fbx"))
    export_gltf(os.path.join(EXPORT_DIR, "character_rigged.glb"))

    # Also save .blend file
    blend_path = os.path.join(EXPORT_DIR, "character_rigged.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"  Blend: {blend_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
