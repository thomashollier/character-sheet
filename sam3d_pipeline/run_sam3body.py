"""Run SAM 3D Body on Modal to get a 3D skeleton from a character image.

Usage:
    modal run run_sam3body.py --image-path /path/to/front_eyelevel.png

Returns 3D joint positions, body mesh, and SMPL parameters.
"""

import modal
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIANGLE_DIR = os.path.dirname(BASE_DIR)  # sam3d_pipeline lives inside multiangle/

# Build the container image with SAM 3D Body
sam3body_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
                 "libegl1-mesa", "libgles2-mesa", "libosmesa6", "mesa-utils")
    .env({"PYOPENGL_PLATFORM": "egl"})
    .pip_install(
        "torch>=2.0",
        "torchvision",
        "pytorch-lightning",
        "opencv-python-headless",
        "yacs",
        "scikit-image",
        "einops",
        "timm",
        "dill",
        "pandas",
        "rich",
        "hydra-core",
        "pyrootutils",
        "roma",
        "joblib",
        "loguru",
        "optree",
        "fvcore",
        "pycocotools",
        "huggingface_hub",
        "networkx==3.2.1",
        "xtcocotools",
        "numpy<2",
        "braceexpand",
        "webdataset",
        "chumpy",
        "smplx",
        "pyrender",
        "trimesh",
        "cloudpickle",
        "omegaconf",
        "iopath",
        "Pillow",
        "matplotlib",
        "black",
        "portalocker",
    )
    # Install detectron2 (needed for ViTDet human detector)
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps"
    )
    # Install MoGe (FOV estimator) and SAM 3D Body
    .pip_install("git+https://github.com/microsoft/MoGe.git")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam-3d-body",
    )
    .env({"PYTHONPATH": "/opt/sam-3d-body"})
    # Download model checkpoint from HuggingFace
    .env({"HF_HOME": "/cache/huggingface"})
)

app = modal.App("sam-3d-body", image=sam3body_image)

# Volume to cache the HF model downloads
model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": model_cache},
)
def run_sam3body(image_bytes: bytes, filename: str) -> tuple[dict, bytes]:
    """Run SAM 3D Body inference on a single image."""
    import torch
    import cv2
    import numpy as np
    import tempfile
    import os
    import sys

    sys.path.insert(0, "/opt/sam-3d-body")

    # Save input image to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        input_path = f.name

    # Read image
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        return {"error": f"Failed to read image: {filename}"}

    print(f"Image loaded: {image_bgr.shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Download checkpoint if not cached
    from huggingface_hub import hf_hub_download, snapshot_download
    checkpoint_dir = "/cache/sam-3d-body-dinov3"

    if not os.path.exists(os.path.join(checkpoint_dir, "model.ckpt")):
        print("Downloading SAM 3D Body checkpoint...")
        snapshot_download(
            "facebook/sam-3d-body-dinov3",
            local_dir=checkpoint_dir,
        )
        print("Download complete.")

    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    mhr_path = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")

    # Use the notebook API to run inference
    os.chdir("/opt/sam-3d-body")
    from notebook.utils import setup_sam_3d_body

    print("Setting up SAM 3D Body estimator...")
    # setup_sam_3d_body takes hf_repo_id or local_dir
    # Since we've downloaded to checkpoint_dir, try local_dir first
    try:
        estimator = setup_sam_3d_body(local_dir=checkpoint_dir)
    except TypeError:
        # Try hf_repo_id approach (downloads from HF)
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    print("Running inference...")
    # Try with output_format parameter for direct FBX/glTF export
    try:
        outputs = estimator.process_one_image(image_bgr, output_format="gltf")
        print(f"Inference complete (gltf mode). Output type: {type(outputs)}")
    except TypeError:
        outputs = estimator.process_one_image(image_bgr)
        print(f"Inference complete (default mode). Output type: {type(outputs)}")

    # Debug: print what we got
    if isinstance(outputs, dict):
        print(f"Output keys: {list(outputs.keys())}")
        for k, v in outputs.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: list len={len(v)}")
            else:
                print(f"  {k}: {type(v).__name__}")
    elif isinstance(outputs, list):
        print(f"Output is list of {len(outputs)} items")
        if outputs:
            item = outputs[0]
            if isinstance(item, dict):
                print(f"  First item keys: {list(item.keys())}")
                for k, v in item.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: shape={v.shape}")
                    else:
                        print(f"    {k}: {type(v).__name__}")

    # Extract the first person's data
    if isinstance(outputs, list):
        person = outputs[0] if outputs else {}
    else:
        person = outputs

    result = {
        "filename": filename,
        "image_shape": list(image_bgr.shape),
    }
    for key, val in person.items():
        if val is None:
            continue
        if hasattr(val, 'cpu'):
            val = val.cpu().numpy()
        if hasattr(val, 'tolist'):
            result[key] = val.tolist()
        elif isinstance(val, (int, float, str, list, dict)):
            result[key] = val

    # --- Build rigged GLB from MHR model ---
    print("\nBuilding rigged mesh from MHR model...")
    mhr_glb = b""
    glb_bytes = b""
    try:
        import trimesh

        verts = np.array(person["pred_vertices"])
        if hasattr(verts, 'cpu'):
            verts = verts.cpu().numpy()

        # Load MHR model to get face topology and skinning
        mhr_data = torch.load(mhr_path, map_location="cpu", weights_only=False)

        # Try to find faces in MHR model
        faces = None
        skin_weights = None
        joint_names = None
        parent_indices = None

        if isinstance(mhr_data, dict):
            print(f"  MHR keys: {list(mhr_data.keys())[:20]}")
            for key in ["faces", "f", "triangles", "tri"]:
                if key in mhr_data:
                    faces = mhr_data[key]
                    if hasattr(faces, 'numpy'):
                        faces = faces.numpy()
                    faces = np.array(faces).astype(np.int32)
                    print(f"  Found faces in '{key}': {faces.shape}")
                    break

            for key in ["weights", "skinning_weights", "lbs_weights", "W"]:
                if key in mhr_data:
                    skin_weights = mhr_data[key]
                    if hasattr(skin_weights, 'numpy'):
                        skin_weights = skin_weights.numpy()
                    print(f"  Found skin weights in '{key}': {np.array(skin_weights).shape}")
                    break

            for key in ["joint_names", "bone_names", "skeleton_names"]:
                if key in mhr_data:
                    joint_names = mhr_data[key]
                    print(f"  Found joint names: {len(joint_names)} joints")
                    break

            for key in ["parents", "parent_indices", "kintree_table"]:
                if key in mhr_data:
                    parent_indices = mhr_data[key]
                    if hasattr(parent_indices, 'numpy'):
                        parent_indices = parent_indices.numpy()
                    print(f"  Found parent indices in '{key}'")
                    break
        else:
            # MHR model is a TorchScript module — probe its buffers and parameters
            print(f"  MHR type: {type(mhr_data).__name__}")

            # Check named buffers (faces are usually stored as buffers)
            for name, buf in mhr_data.named_buffers():
                shape = tuple(buf.shape) if hasattr(buf, 'shape') else '?'
                # Only match mesh.faces specifically (Nx3 int tensor)
                if name.endswith("mesh.faces") and len(buf.shape) == 2 and buf.shape[1] == 3:
                    faces = buf.cpu().numpy().astype(np.int32)
                    print(f"  FACES: {name}: {shape}")
                if name.endswith("joint_parents") and parent_indices is None:
                    parent_indices = buf.cpu().numpy()
                    print(f"  PARENTS: {name}: {shape}")
                if name.endswith("skin_weights_flattened") and skin_weights is None:
                    skin_weights = buf.cpu().numpy()
                    skin_indices = None
                    # Also grab skin indices
                    for n2, b2 in mhr_data.named_buffers():
                        if n2.endswith("skin_indices_flattened"):
                            skin_indices = b2.cpu().numpy()
                            break
                    for n2, b2 in mhr_data.named_buffers():
                        if n2.endswith("vert_indices_flattened"):
                            vert_indices = b2.cpu().numpy()
                            break
                    print(f"  SKINNING: {name}: {shape}")
                if name.endswith("joint_translation_offsets"):
                    joint_offsets = buf.cpu().numpy()
                    print(f"  JOINT OFFSETS: {name}: {shape}")

            # Check named parameters
            for name, param in mhr_data.named_parameters():
                shape = tuple(param.shape) if hasattr(param, 'shape') else '?'
                print(f"  param: {name}: {shape}")

            # Also check the estimator for face data
            if hasattr(estimator, 'mhr_model'):
                mhr = estimator.mhr_model
                for name, buf in mhr.named_buffers():
                    shape = tuple(buf.shape)
                    if "face" in name.lower() and faces is None:
                        faces = buf.cpu().numpy().astype(np.int32)
                        print(f"  estimator.mhr_model buffer: {name}: {shape}")

            # Check sub-modules
            for name, mod in mhr_data.named_modules():
                if name:  # skip root
                    print(f"  module: {name}: {type(mod).__name__}")
                    for bname, buf in mod.named_buffers(recurse=False):
                        shape = tuple(buf.shape)
                        print(f"    buffer: {bname}: {shape} {buf.dtype}")
                        if "face" in bname.lower() and faces is None:
                            faces = buf.cpu().numpy().astype(np.int32)

        if faces is not None:
            print(f"  Building mesh: {len(verts)} verts, {len(faces)} faces")

            # Store faces and skinning data in result for local export
            result["mesh_faces_data"] = faces.tolist()
            result["mesh_faces"] = len(faces)
            if parent_indices is not None:
                result["joint_parents"] = parent_indices.tolist()
            if skin_weights is not None:
                result["skin_weights_flat"] = skin_weights.tolist()
            if skin_indices is not None:
                result["skin_indices_flat"] = skin_indices.tolist()
            if vert_indices is not None:
                result["vert_indices_flat"] = vert_indices.tolist()
            if 'joint_offsets' in dir() and joint_offsets is not None:
                result["joint_offsets"] = joint_offsets.tolist()

            # Export mesh as OBJ (with faces, universally compatible)
            obj_lines = [f"# SAM 3D Body mesh: {len(verts)} verts, {len(faces)} faces\n"]
            for v in verts:
                obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f in faces:
                obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
            obj_bytes = "".join(obj_lines).encode()

            # Export as GLB (mesh without skeleton for quick viewing)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            glb_bytes = mesh.export(file_type="glb")
            print(f"  GLB size: {len(glb_bytes) / 1024:.0f} KB")
            print(f"  OBJ size: {len(obj_bytes) / 1024:.0f} KB")

            # Store OBJ too
            result["_obj_bytes_b64"] = __import__("base64").b64encode(obj_bytes).decode()
        else:
            print("  No face topology found in MHR model")
            mesh = trimesh.Trimesh(vertices=verts, process=False)
            glb_bytes = mesh.export(file_type="glb")

    except Exception as e:
        print(f"  GLB export failed: {e}")
        import traceback
        traceback.print_exc()

    os.unlink(input_path)
    # Prefer MHR-exported mesh if available, otherwise use our GLB
    final_mesh = mhr_glb if mhr_glb else glb_bytes
    return result, final_mesh


@app.local_entrypoint()
def main(image_path: str = ""):
    """Run SAM 3D Body on an image and save results locally."""
    if not image_path:
        # Default to the front eyelevel image
        candidates = [
            os.path.join(MULTIANGLE_DIR, "pose_full_output",
                         "az000_el+00_d1.0_front_view_eyelevel_shot_medium_shot.png"),
            os.path.join(MULTIANGLE_DIR, "pose_full_output",
                         "az000_el+00_d1.8_front_view_eyelevel_shot_wide_shot.png"),
        ]
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                break

    if not image_path or not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Usage: modal run run_sam3body.py --image-path /path/to/image.png")
        sys.exit(1)

    print(f"Processing: {image_path}")
    filename = os.path.basename(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Image size: {len(image_bytes) / 1024:.0f} KB")
    print("Sending to Modal...")

    result, glb_bytes = run_sam3body.remote(image_bytes, filename)

    # Save JSON result
    out_path = os.path.join(BASE_DIR, "sam3body_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {out_path}")

    # Save exports
    export_dir = os.path.join(BASE_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)

    if glb_bytes:
        glb_path = os.path.join(export_dir, "character_mesh.glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        print(f"Saved GLB: {glb_path} ({len(glb_bytes) / 1024:.0f} KB)")

    # Save OBJ (mesh with faces — opens in any DCC)
    if "_obj_bytes_b64" in result:
        import base64
        obj_bytes = base64.b64decode(result["_obj_bytes_b64"])
        obj_path = os.path.join(export_dir, "character_mesh.obj")
        with open(obj_path, "wb") as f:
            f.write(obj_bytes)
        print(f"Saved OBJ: {obj_path} ({len(obj_bytes) / 1024:.0f} KB)")
        del result["_obj_bytes_b64"]  # don't save in JSON

    # Print summary
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\n=== Results ===")
    print(f"  3D keypoints:  {len(result.get('pred_keypoints_3d', []))} joints")
    print(f"  Joint coords:  {len(result.get('pred_joint_coords', []))} joints")
    print(f"  Mesh vertices: {len(result.get('pred_vertices', []))}")
    print(f"  Mesh faces:    {result.get('mesh_faces', 0)}")
    print(f"  Has skeleton:  {'joint_parents' in result}")
    print(f"  Has skinning:  {'skin_weights_flat' in result}")

    # Save clean JSON (without huge arrays, those go in a separate file)
    result_slim = {k: v for k, v in result.items()
                   if k not in ("pred_vertices", "mesh_faces_data",
                                "skin_weights_flat", "skin_indices_flat",
                                "vert_indices_flat")}
    slim_path = os.path.join(BASE_DIR, "sam3body_result.json")
    with open(slim_path, "w") as f:
        json.dump(result_slim, f, indent=2)
    print(f"\nSaved skeleton JSON: {slim_path}")

    # Save full data (with mesh + skinning) as separate file
    full_path = os.path.join(export_dir, "character_full_data.json")
    with open(full_path, "w") as f:
        json.dump(result, f)
    print(f"Saved full data: {full_path} ({os.path.getsize(full_path) / 1024:.0f} KB)")
