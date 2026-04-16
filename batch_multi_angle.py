#!/usr/bin/env python3
"""
Qwen Image Edit — Multi-Angle Batch Renderer (ComfyUI API)
================================================================
Renders all 96 camera poses (8 azimuths × 4 elevations × 3 distances)
from a single input image. Supports both 2509 and 2511 pipelines.

Supports both LOCAL ComfyUI and COMFY CLOUD (cloud.comfy.org).

Requirements:
  pip install aiohttp   (only needed for --cloud mode)

Usage:
  # ── 2511 pipeline (default) ────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud

  # ── 2509 pipeline ──────────────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud --pipeline 2509

  # ── SUBSET / DRY RUN ───────────────────────────────────────
  python batch_multi_angle.py --image photo.png --cloud --dry-run
  python batch_multi_angle.py --image photo.png --cloud --azimuths 0,90,180,270
"""

import argparse
import json
import os
import sys
import time
import tempfile
import urllib.request
import urllib.parse
import uuid
import asyncio
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# ANGLE DEFINITIONS (from the LoRA training data)
# ═══════════════════════════════════════════════════════════════════════════

AZIMUTH_MAP = {
    0:   "front view",
    45:  "front-right quarter view",
    90:  "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}

ELEVATION_MAP = {
    -30: "low-angle shot",
    0:   "eye-level shot",
    30:  "elevated shot",
    60:  "high-angle shot",
}

DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot",
}

ALL_AZIMUTHS   = sorted(AZIMUTH_MAP.keys())
ALL_ELEVATIONS = sorted(ELEVATION_MAP.keys())
ALL_DISTANCES  = sorted(DISTANCE_MAP.keys())


def build_prompt_2511(azimuth, elevation, distance):
    return f"<sks> {AZIMUTH_MAP[azimuth]} {ELEVATION_MAP[elevation]} {DISTANCE_MAP[distance]}"


def build_prompt_2509(azimuth, elevation, distance):
    """Build bilingual Chinese+English camera prompt for the 2509 angles LoRA."""
    parts = []
    # Rotation
    if azimuth != 0:
        if azimuth <= 180:
            parts.append(
                f"将镜头向右旋转{azimuth}度 "
                f"Rotate the camera {azimuth} degrees to the right."
            )
        else:
            deg = 360 - azimuth
            parts.append(
                f"将镜头向左旋转{deg}度 "
                f"Rotate the camera {deg} degrees to the left."
            )
    # Elevation / vertical tilt
    if elevation < 0:
        parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")
    elif elevation > 0:
        parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    # Distance / zoom
    if distance < 0.8:
        parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif distance > 1.5:
        parts.append("将镜头向后移动 Move the camera backward.")
    return " ".join(parts) if parts else "no camera movement"


def safe_filename(azimuth, elevation, distance):
    az_name   = AZIMUTH_MAP[azimuth].replace(" ", "_").replace("-", "")
    el_name   = ELEVATION_MAP[elevation].replace(" ", "_").replace("-", "")
    dist_name = DISTANCE_MAP[distance].replace("-", "").replace(" ", "_")
    return f"az{azimuth:03d}_el{elevation:+03d}_d{distance:.1f}_{az_name}_{el_name}_{dist_name}.png"


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSIONS
# ═══════════════════════════════════════════════════════════════════════════

EXPRESSIONS = {
    "neutral":       "Change to a calm, neutral expression with a relaxed face and loose, natural posture.",
    "happy":         "Change to a genuinely happy smile with bright eyes, shoulders lifted slightly and head tilted with warmth.",
    "laughing":      "Change to laughing out loud with an open mouth, squinted eyes, head thrown back slightly and shoulders shaking.",
    "smirk":         "Change to a subtle smirk with one corner of the mouth raised, chin tilted down slightly with a knowing look.",
    "sad":           "Change to a sad, sorrowful look with downturned mouth and drooping eyes, shoulders slumped and head bowed slightly.",
    "crying":        "Change to crying with tears streaming down the cheeks, face scrunched in pain, hands raised near the face.",
    "angry":         "Change to an angry scowl with furrowed brows and clenched jaw, shoulders squared and tensed, leaning forward slightly.",
    "disgusted":     "Change to a look of disgust with a wrinkled nose and curled upper lip, head pulled back and turned slightly away.",
    "surprised":     "Change to wide-eyed surprise with raised eyebrows and open mouth, shoulders pulled up and body leaning back slightly.",
    "fearful":       "Change to a fearful, scared look with wide eyes and tense mouth, shoulders hunched and body shrinking back.",
    "confused":      "Change to a confused look with one raised eyebrow and a slight frown, head tilted to the side.",
    "determined":    "Change to a fierce, determined look with a set jaw and focused eyes, chin raised and chest forward.",
    "flirty":        "Change to a playful, mischievous look with a warm smile and slightly raised eyebrow, head tilted gently.",
    "contempt":      "Change to a contemptuous facial expression with a slight sneer and raised chin, keep the same background.",
    "embarrassed":   "Change to an embarrassed blush with averted gaze and a sheepish smile, shoulders drawn in and head ducked.",
    "sleepy":        "Change to a neutral relaxed expression but lower the eyelids about a third of the way closed. Keep eyebrows in their natural position, just slightly relaxed. Closed mouth with barely upturned lip corners, fully relaxed jaw, serene surrendered expression.",
}

LIGHTING = {
    "rim_light":     "Change the lighting so a bright light source is directly behind the figure, creating a glowing outline around the hair and shoulders. The front of the face is darker with only ambient fill light.",
    "side_light":    "Change the lighting to strong directional side lighting from the left, casting deep shadows on the right side of the face and body.",
    "golden_hour":   "Change the lighting to warm golden hour sunlight, with long soft shadows and a warm orange glow on the skin.",
    "moonlight":     "Change the lighting to cool blue moonlight at night, with soft shadows and a pale silvery glow on the face.",
}

LIGHTING_PREFIX = (
    "Keep the same pose, outfit, hairstyle, hair color, identity, and facial expression. "
    "Only change the lighting and shadows. "
)

OUTFITS = {
    "formal":        "Change the outfit to a simple nice sundress that is a little too fancy for her, keep the same facial expression, pose, and background.",
    "athletic":      "Change the outfit to a sporty soccer jersey with short tight sleeves and a team crest on the chest, keep the same pose and background.",
    "winter":        "Change the outfit to a cozy winter outfit with a knit sweater, scarf, and warm jacket, keep the same pose and background.",
    "work":          "Change the outfit to a plant nursery work apron over a green t-shirt, with soil stains on the apron, a small plant seedling poking out of the front pocket, and a hand trowel tucked in the left side of the apron. A small name tag pinned to the upper right of the apron. Do not change the background, keep the original plain background.",
}

OUTFITS_PREFIX = (
    "Keep the same pose, hairstyle, hair color, identity, facial expression, and background. "
    "Only change the clothing. "
)

POSES = {
    "running":        "Change the pose to running at full speed. Left leg extended forward, right leg pushing off behind. Arms bent at elbows pumping in opposite rhythm to legs. Torso leaning forward, hair flowing back.",
    "jumping_joy":    "Change the pose to jumping in the air with joy. Both feet off the ground, knees bent. Both arms raised high above the head, fingers spread wide. Head tilted up, back slightly arched.",
    "sitting_cross":  "Change the pose to sitting cross-legged on the ground. Legs folded with ankles crossed, hands resting on knees. Back straight, shoulders relaxed, head level.",
    "waving":         "Change the pose to waving hello. Right arm raised high above the shoulder, hand open with fingers together, wrist tilted side to side. Left arm relaxed at the side. Weight on both feet, slight lean toward the viewer.",
    "thinking":       "Change the pose to deep in thought. Right hand raised to chin, index finger touching the lower lip. Left arm crossed under the right elbow. Weight shifted to left leg, right foot slightly forward. Head tilted slightly to the right, eyes looking up.",
    "dancing":        "Change the pose to dancing energetically. Right arm extended out to the side at shoulder height, left arm bent with hand near the face. Right leg stepping to the side with pointed toe, left leg bent at the knee. Hips shifted to the left, torso twisting.",
    "walking":        "Change the pose to walking at a casual pace, viewed from a 3/4 angle. Left leg stepping forward, right leg behind mid-stride. Left arm swinging back, right arm swinging forward naturally. Torso upright, head facing the walking direction.",
    "crouching":      "Change the pose to crouching down low. Knees deeply bent, body low to the ground. Arms resting on top of the knees, hands dangling. Head ducked forward, looking up from below.",
    "pointing":       "Change the pose to pointing at something in the distance. Right arm fully extended forward, index finger pointing. Left hand on hip. Body turned slightly to the right, weight on the right foot, left foot angled out.",
    "hands_on_hips":  "Change the pose to standing confidently with both hands on hips. Elbows pushed out wide, fingers wrapped around the waist. Feet shoulder-width apart, chest out, chin raised slightly.",
    "kicking":        "Change the pose to kicking a ball. Right leg swung forward and up at hip height, toes pointed. Left leg planted firmly, slightly bent. Arms out to the sides for balance. Torso leaning back slightly.",
    "jump_rope":      "Change the pose to jumping rope, viewed from a slight 3/4 angle. She is at the peak of her jump with both feet off the ground, one knee higher than the other. The rope passing beneath her feet. Body slightly turned, not perfectly symmetrical. Hair flying up, joyful expression.",
    "talking":        "Change the pose to standing at a 3/4 angle, in confident conversation. One hand raised at chest height with palm up, making a calm deliberate gesture. Other hand relaxed at her side. Weight on one leg, posture upright and poised. Calm confident expression, slight smile, speaking with composure.",
    "leaning_wall":   "Change the pose to leaning against a wall. Right shoulder pressed against the wall, arms crossed over the chest. Right leg crossed over the left at the ankle. Head tilted slightly, looking forward with a relaxed expression.",
    "reading":        "Change the pose to sitting on an institutional mid-century light colored wood chair, reading a book, viewed from a slight 3/4 right angle. Legs crossed at the ankles, leaning back comfortably. Both hands holding an open book at chest height, head tilted down looking at the pages. Relaxed, absorbed expression.",
    "shy_stance":     "Change the pose to a shy, bashful stance. Hands clasped together in front of the body at waist level, fingers intertwined. Shoulders raised slightly, one foot turned inward pigeon-toed. Head tilted down with eyes looking up.",
}

ANGLES_PROMPT = {}
_GAZE_BRAID = "The character's eyes and gaze face the same direction the body is pointing. The braids drape in front of the shoulders, not behind."
_DIRECTIONS = [
    ("front",       "000", "front-facing",                                          ""),
    ("front_right", "045", "from a 3/4 front-right angle",                          "Rotate the camera 45 degrees to the right."),
    ("right",       "090", "in right side profile",                                 "Rotate the camera 90 degrees to the right."),
    ("back_right",  "135", "from behind and slightly to the right",                 "Rotate the camera 135 degrees to the right."),
    ("back",        "180", "from directly behind",                                  "Rotate the camera 180 degrees to show the back."),
    ("back_left",   "225", "from behind and slightly to the left",                  "Rotate the camera 135 degrees to the left."),
    ("left",        "270", "in left side profile",                                  "Rotate the camera 90 degrees to the left."),
    ("front_left",  "315", "from a 3/4 front-left angle",                           "Rotate the camera 45 degrees to the left."),
]
for _name, _deg, _view, _rotate in _DIRECTIONS:
    _r = f"{_rotate} " if _rotate else ""
    ANGLES_PROMPT[f"{_name}_close"]  = f"{_r}Close-up shot showing head and shoulders {_view}. {_GAZE_BRAID}"
    ANGLES_PROMPT[f"{_name}_medium"] = f"{_r}Move the camera much closer to the subject. The image should only contain the upper half of the body, from head to waist {_view}. The lower body is cut off by the bottom of the frame. {_GAZE_BRAID}"
    ANGLES_PROMPT[f"{_name}_wide"]   = f"{_r}Full body wide shot {_view}, showing the entire figure from head to toe. {_GAZE_BRAID}"
    # Elevated 30° versions (camera above looking down)
    _elev_cam = "The camera is higher up, at head height plus 3 feet, angled slightly downward. The horizon line is low in the frame. Do not roll or rotate the camera, keep it perfectly level left to right."
    _neutral_gaze = "Keep the exact same neutral facial expression as the original image. The character looks straight ahead in the direction the body faces, not at the camera. The braids drape in front of the shoulders, not behind."
    ANGLES_PROMPT[f"{_name}_elev_close"]  = f"{_r}{_elev_cam} Close-up shot showing head and shoulders {_view}. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_elev_medium"] = f"{_r}{_elev_cam} Move the camera much closer. The image should only contain the upper half of the body from head to waist {_view}. The lower body is cut off by the bottom of the frame. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_elev_wide"]   = f"{_r}{_elev_cam} Full body wide shot {_view}, showing the entire figure from head to toe. {_neutral_gaze}"
    # Low angle 30° versions (camera below looking up)
    _low_cam = "The camera is lower, at knee height, angled slightly upward. The horizon line is high in the frame. Do not roll or rotate the camera, keep it perfectly level left to right."
    ANGLES_PROMPT[f"{_name}_low_close"]  = f"{_r}{_low_cam} Close-up shot showing head and shoulders {_view}. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_low_medium"] = f"{_r}{_low_cam} Move the camera much closer. The image should only contain the upper half of the body from head to waist {_view}. The lower body is cut off by the bottom of the frame. {_neutral_gaze}"
    ANGLES_PROMPT[f"{_name}_low_wide"]   = f"{_r}{_low_cam} Full body wide shot {_view}, showing the entire figure from head to toe. {_neutral_gaze}"

ANGLES_PROMPT_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, identity, and the same plain solid color background. "
    "Do not change the background color or add any scene elements. "
)

POSES_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, identity, and the same plain solid color background. "
    "Do not change the background color or add any scene elements. "
    "Clothing should move naturally with the body. "
)

EXPRESSION_PREFIX = (
    "Keep the same outfit, hairstyle, hair color, background, lighting, "
    "and identity. Clothing should move naturally with the body. "
)


# ═══════════════════════════════════════════════════════════════════════════
# ANYPOSE DEFAULT PROMPT
# ═══════════════════════════════════════════════════════════════════════════

ANYPOSE_DEFAULT_PROMPT = (
    "Make the person in image 1 do the exact same pose of the person in image 2. "
    "Changing the style and background of the image of the person in image 1 is undesirable, so don't do it. "
    "The new pose should be pixel accurate to the pose we are trying to copy. "
    "The position of the arms and head and legs should be the same as the pose we are trying to copy. "
    "Change the field of view and angle to match exactly image 2. Head tilt and eye gaze pose should match the person in image 2. "
    "Remove the background of image 2, and replace it with the background of image 1. "
    "Don't change the identity of the person in image 1, keep their appearance the same, "
    "it is undesirable to change their facial features or hair style. don't do it."
)


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_workflow(
    image_filename, azimuth, elevation, distance, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle", pipeline="2511",
    pose_image_filename=None,
):
    """Dispatch to the appropriate pipeline workflow builder."""
    if pipeline == "2509":
        return build_workflow_2509(
            image_filename, prompt, seed, steps, guidance_scale,
            lora_strength_lightning, lora_strength_angles, filename_prefix,
        )
    if pipeline == "anypose":
        return build_workflow_anypose(
            image_filename, pose_image_filename, prompt, seed, steps,
            guidance_scale, lora_strength_lightning, filename_prefix,
        )
    if pipeline in ("expressions", "lighting", "outfits", "poses_prompt", "angles_prompt"):
        return build_workflow_expressions(
            image_filename, prompt, seed, steps, guidance_scale,
            lora_strength_lightning, filename_prefix,
        )
    return build_workflow_2511(
        image_filename, azimuth, elevation, distance,
        seed, steps, guidance_scale,
        lora_strength_lightning, lora_strength_angles, filename_prefix,
    )


def build_workflow_2511(
    image_filename, azimuth, elevation, distance,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle",
):
    """
    Build a ComfyUI API-format workflow for 2511 pipeline.
    Uses QwenMultiangleCameraNode for camera control.
    """
    return {
        # ── Load input image ──────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["4:102", 0],
            },
        },
        # ── Camera angle node (generates prompt automatically) ────
        "3": {
            "class_type": "QwenMultiangleCameraNode",
            "inputs": {
                "horizontal_angle": azimuth,
                "vertical_angle": elevation,
                "zoom": distance,
                "default_prompts": False,
                "camera_view": False,
                "image": ["1", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "4:95": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "4:100": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["4:93", 0],
                "vae": ["4:95", 0],
                "image1": ["4:106", 0],
            },
        },
        # ── Positive conditioning (prompt from camera node) ───────
        "4:103": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": ["3", 0],
                "clip": ["4:93", 0],
                "vae": ["4:95", 0],
                "image1": ["4:106", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "4:97": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["4:103", 0],
            },
        },
        "4:96": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["4:100", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "4:94": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["4:110", 0],
            },
        },
        "4:98": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["4:94", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "4:104": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["4:106", 0],
                "vae": ["4:95", 0],
            },
        },
        "4:102": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["4:105", 0],
                "vae": ["4:95", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "4:105": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["4:98", 0],
                "positive": ["4:97", 0],
                "negative": ["4:96", 0],
                "latent_image": ["4:104", 0],
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "4:108": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        "4:107": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["4:108", 0],
            },
        },
        "4:110": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
                "strength_model": lora_strength_angles,
                "model": ["4:107", 0],
            },
        },
        # ── CLIP ──────────────────────────────────────────────────
        "4:93": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "4:106": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
    }


def build_workflow_2509(
    image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, lora_strength_angles=1.0,
    filename_prefix="multi_angle",
):
    """
    Build a ComfyUI API-format workflow for 2509 pipeline.
    Based on the 1-click multiple character angles template.
    """
    return {
        # ── Load input image ──────────────────────────────────────
        "25": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "31": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["162:19", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "48:9": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "48:10": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "48:12": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "48:26": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["48:12", 0],
            },
        },
        # ── Angles LoRA ───────────────────────────────────────────
        "48:20": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Edit-2509-Multiple-angles.safetensors",
                "strength_model": lora_strength_angles,
                "model": ["48:26", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "162:11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.0,
                "model": ["48:20", 0],
            },
        },
        "162:8": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["162:11", 0],
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "162:28": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "upscale_method": "nearest-exact",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": ["25", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "162:14": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["48:10", 0],
                "vae": ["48:9", 0],
                "image1": ["162:28", 0],
            },
        },
        # ── Positive conditioning ─────────────────────────────────
        "162:17": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["48:10", 0],
                "vae": ["48:9", 0],
                "image1": ["162:28", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "162:13": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["162:28", 0],
                "vae": ["48:9", 0],
            },
        },
        "162:19": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["162:21", 0],
                "vae": ["48:9", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "162:21": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["162:8", 0],
                "positive": ["162:17", 0],
                "negative": ["162:14", 0],
                "latent_image": ["162:13", 0],
            },
        },
    }


def build_workflow_expressions(
    image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, filename_prefix="expression",
):
    """
    Build a ComfyUI API-format workflow for expression editing.
    Uses 2511 base model + Lightning LoRA with a text prompt.
    """
    return {
        # ── Load input image ──────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["9", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "21": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "30": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "31": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["30", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "34": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["31", 0],
            },
        },
        "35": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["34", 0],
            },
        },
        # ── Image scaling ─────────────────────────────────────────
        "40": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
        # ── Positive conditioning ─────────────────────────────────
        "50": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "51": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "52": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["50", 0],
            },
        },
        "53": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["51", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "60": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["20", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["70", 0],
                "vae": ["20", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "70": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["35", 0],
                "positive": ["52", 0],
                "negative": ["53", 0],
                "latent_image": ["60", 0],
            },
        },
    }


def build_workflow_anypose(
    image_filename, pose_image_filename, prompt,
    seed=42, steps=4, guidance_scale=1.0,
    lora_strength_lightning=1.0, filename_prefix="anypose",
    lora_strength_base=0.7, lora_strength_helper=0.7,
):
    """
    Build a ComfyUI API-format workflow for AnyPose pipeline.
    Takes a reference image and a pose image, transfers the pose.
    Uses 2511 base model + Lightning + AnyPose base + AnyPose helper LoRAs.
    """
    return {
        # ── Load reference image ──────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # ── Load pose image ───────────────────────────────────────
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image_filename},
        },
        # ── Save output ───────────────────────────────────────────
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["9", 0],
            },
        },
        # ── VAE ───────────────────────────────────────────────────
        "20": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        # ── CLIP ──────────────────────────────────────────────────
        "21": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        # ── Model loading ─────────────────────────────────────────
        "30": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2511_bf16.safetensors",
                "weight_dtype": "default",
            },
        },
        # ── Lightning LoRA ────────────────────────────────────────
        "31": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": lora_strength_lightning,
                "model": ["30", 0],
            },
        },
        # ── AnyPose base LoRA ─────────────────────────────────────
        "32": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "2511-AnyPose-base-000006250.safetensors",
                "strength_model": lora_strength_base,
                "model": ["31", 0],
            },
        },
        # ── AnyPose helper LoRA ───────────────────────────────────
        "33": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "2511-AnyPose-helper-00006000.safetensors",
                "strength_model": lora_strength_helper,
                "model": ["32", 0],
            },
        },
        # ── Model patching ────────────────────────────────────────
        "34": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.1,
                "model": ["33", 0],
            },
        },
        "35": {
            "class_type": "CFGNorm",
            "inputs": {
                "strength": guidance_scale,
                "model": ["34", 0],
            },
        },
        # ── Image scaling (reference) ─────────────────────────────
        "40": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["1", 0],
            },
        },
        # ── Image scaling (pose) ──────────────────────────────────
        "41": {
            "class_type": "FluxKontextImageScale",
            "inputs": {
                "image": ["2", 0],
            },
        },
        # ── Positive conditioning (reference + pose images) ───────
        "50": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": prompt,
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
                "image2": ["41", 0],
            },
        },
        # ── Negative conditioning (empty prompt) ──────────────────
        "51": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "",
                "clip": ["21", 0],
                "vae": ["20", 0],
                "image1": ["40", 0],
            },
        },
        # ── Reference latent methods ──────────────────────────────
        "52": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["50", 0],
            },
        },
        "53": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "reference_latents_method": "index_timestep_zero",
                "conditioning": ["51", 0],
            },
        },
        # ── VAE Encode / Decode ───────────────────────────────────
        "60": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["20", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["70", 0],
                "vae": ["20", 0],
            },
        },
        # ── KSampler ─────────────────────────────────────────────
        "70": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": guidance_scale,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["35", 0],
                "positive": ["52", 0],
                "negative": ["53", 0],
                "latent_image": ["60", 0],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL COMFYUI API  (synchronous, sequential)
# ═══════════════════════════════════════════════════════════════════════════

def local_upload_image(server_url, filepath):
    filepath = Path(filepath)
    boundary = uuid.uuid4().hex
    with open(filepath, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filepath.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{server_url}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["name"]


def local_queue(server_url, workflow, client_id):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{server_url}/prompt", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["prompt_id"]


def local_wait(server_url, prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{server_url}/history/{prompt_id}") as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out after {timeout}s")


def local_download(server_url, filename, subfolder, output_path):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    urllib.request.urlretrieve(f"{server_url}/view?{params}", output_path)


def run_local(jobs, args):
    print(f"  Mode: LOCAL → {args.server}\n")
    print("Uploading image...")
    try:
        server_filename = local_upload_image(args.server, args.image)
    except Exception as e:
        sys.exit(f"ERROR: Could not reach ComfyUI at {args.server}\n  {e}")
    print(f"  Uploaded as: {server_filename}\n")

    client_id = str(uuid.uuid4())
    total, ok, fail = len(jobs), 0, 0

    for i, (az, el, dist, prompt, fname) in enumerate(jobs, 1):
        out = os.path.join(args.output, fname)
        if os.path.exists(out):
            print(f"  [{i:3d}/{total}] SKIP  {fname}")
            ok += 1; continue
        print(f"  [{i:3d}/{total}] {prompt}")
        print(f"           → {fname}", end="", flush=True)
        try:
            wf = build_workflow(server_filename, az, el, dist, prompt,
                                args.seed, args.steps,
                                args.guidance, args.lora_lightning, args.lora_angles,
                                pipeline=args.pipeline)
            pid = local_queue(args.server, wf, client_id)
            result = local_wait(args.server, pid, args.timeout)
            outputs = result.get("outputs", {})
            save_node = "31" if args.pipeline == "2509" else "2"
            images = outputs.get(save_node, {}).get("images", [])
            if images:
                local_download(args.server, images[0]["filename"],
                               images[0].get("subfolder", ""), out)
                print("  ✓"); ok += 1
            else:
                print("  ✗ no output"); fail += 1
        except Exception as e:
            print(f"  ✗ {e}"); fail += 1
    return ok, fail


# ═══════════════════════════════════════════════════════════════════════════
# COMFY CLOUD API  (async, parallel)
# ═══════════════════════════════════════════════════════════════════════════

CLOUD_BASE = "https://cloud.comfy.org"


def get_reference_bg_color(image_path):
    """Get the average edge color of the reference image."""
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(edges.mean(axis=0).astype(int))


def prep_pose_image(pose_path, bg_color, tmp_dir):
    """Replace black background with bg_color and pad to square."""
    from PIL import Image
    import numpy as np
    img = Image.open(pose_path).convert("RGB")
    arr = np.array(img)
    # Replace black pixels with bg color
    mask = (arr[:, :, 0] < 15) & (arr[:, :, 1] < 15) & (arr[:, :, 2] < 15)
    arr[mask] = bg_color
    img = Image.fromarray(arr)
    # Pad to square if needed
    w, h = img.size
    if w != h:
        size = max(w, h)
        square = Image.new("RGB", (size, size), bg_color)
        square.paste(img, ((size - w) // 2, (size - h) // 2))
        img = square
    out = os.path.join(tmp_dir, os.path.basename(pose_path))
    img.save(out)
    return out


async def cloud_upload(session, api_key, filepath):
    import aiohttp
    filepath = Path(filepath)
    data = aiohttp.FormData()
    data.add_field("image", open(filepath, "rb"),
                   filename=filepath.name, content_type="image/png")
    data.add_field("type", "input")
    data.add_field("overwrite", "true")
    async with session.post(f"{CLOUD_BASE}/api/upload/image",
                            headers={"X-API-Key": api_key}, data=data) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Upload failed ({resp.status}): {await resp.text()}")
        return (await resp.json())["name"]


async def cloud_submit(session, api_key, workflow):
    async with session.post(
        f"{CLOUD_BASE}/api/prompt",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"prompt": workflow},
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Submit failed ({resp.status}): {await resp.text()}")
        return (await resp.json())["prompt_id"]


def cleanup_checkerboard(output_path, ref_image_path):
    """Replace checkerboard artifacts near the background color with solid bg."""
    try:
        from PIL import Image
        import numpy as np
        ref = Image.open(ref_image_path).convert("RGB")
        arr = np.array(ref)
        edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg = edges.mean(axis=0).astype(int)

        img = Image.open(output_path).convert("RGB")
        a = np.array(img)
        diff = np.abs(a.astype(int) - bg.astype(int)).sum(axis=2)
        a[diff < 60] = bg
        Image.fromarray(a).save(output_path)
    except Exception:
        pass  # non-critical, skip if it fails


async def cloud_download(session, api_key, filename, output_path):
    """Download an output file via /api/view (follows redirect to storage)."""
    params = urllib.parse.urlencode({"filename": filename, "subfolder": "", "type": "output"})
    url = f"{CLOUD_BASE}/api/view?{params}"
    async with session.get(url, headers={"X-API-Key": api_key}, allow_redirects=False) as resp:
        if resp.status in (301, 302, 303, 307, 308):
            # Follow redirect without auth header (storage URL doesn't need it)
            redirect_url = resp.headers["Location"]
            async with session.get(redirect_url) as dl_resp:
                if dl_resp.status != 200:
                    raise RuntimeError(f"Download failed ({dl_resp.status}) from redirect")
                with open(output_path, "wb") as f:
                    f.write(await dl_resp.read())
        elif resp.status == 200:
            with open(output_path, "wb") as f:
                f.write(await resp.read())
        else:
            raise RuntimeError(f"Download failed ({resp.status}): {await resp.text()}")


async def _ws_collect_outputs(api_key, prompt_ids, timeout=600):
    """
    Connect to Comfy Cloud websocket and collect output filenames for given prompt_ids.
    Returns dict: {prompt_id: [{"filename": ..., "subfolder": ..., "type": ...}, ...]}
    """
    import aiohttp
    results = {}
    pending = set(prompt_ids)
    ws_url = f"wss://cloud.comfy.org/ws?token={api_key}"

    async with aiohttp.ClientSession() as ws_session:
        async with ws_session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=10)) as ws:
            start = time.time()
            while pending and (time.time() - start) < timeout:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                if msg.type in (aiohttp.WSMsgType.TEXT,):
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    msg_type = data.get("type", "")
                    msg_data = data.get("data", {})
                    pid = msg_data.get("prompt_id", "")
                    if msg_type == "executed" and pid in pending:
                        # executed message contains output info
                        node_output = msg_data.get("output", {})
                        images = node_output.get("images", [])
                        if images:
                            results[pid] = images
                            pending.discard(pid)
                    elif msg_type == "execution_error" and pid in pending:
                        results[pid] = []
                        pending.discard(pid)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
    return results


async def _process_batch_ws(jobs, args, api_key, session):
    """Submit all jobs, listen on websocket for outputs, download results."""
    import aiohttp
    total = len(jobs)
    sem = asyncio.Semaphore(args.concurrency)
    ok, fail = 0, 0

    # Upload image(s)
    print("Uploading image to Comfy Cloud...")
    try:
        img_name = await cloud_upload(session, api_key, args.image)
    except Exception as e:
        sys.exit(f"ERROR: Upload failed — {e}")
    print(f"  Uploaded as: {img_name}\n")

    # For AnyPose, pre-process and upload all pose images
    pose_img_names = {}  # local_path -> cloud_name
    if args.pipeline == "anypose":
        pose_paths = sorted(set(j[5] for j in jobs))  # jobs[5] = pose_path
        bg_color = get_reference_bg_color(args.image)
        print(f"  Reference bg color: RGB{bg_color}")
        tmp_dir = tempfile.mkdtemp(prefix="anypose_")
        print(f"Uploading {len(pose_paths)} pose images (bg-matched)...")
        for pp in pose_paths:
            try:
                prepped = prep_pose_image(pp, bg_color, tmp_dir)
                pname = await cloud_upload(session, api_key, prepped)
                pose_img_names[pp] = pname
                print(f"  ✓ {os.path.basename(pp)} → {pname}")
            except Exception as e:
                print(f"  ✗ {os.path.basename(pp)} — {e}")
        print()

    pid_to_job = {}  # prompt_id -> (idx, fname, out_path, prompt)
    pending = set()
    ws_url = f"wss://cloud.comfy.org/ws?token={api_key}"

    # Connect websocket FIRST, then submit jobs so we don't miss early completions
    print("  Connecting websocket...")
    async with session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=10),
                                  heartbeat=30) as ws:
        # Wait for initial status message
        try:
            await asyncio.wait_for(ws.receive(), timeout=5)
        except asyncio.TimeoutError:
            pass
        print("  Websocket connected.\n")

        # Submit jobs with concurrency limit
        async def submit_one(idx, job):
            az, el, dist, prompt, fname = job[:5]
            out = os.path.join(args.output, fname)
            if os.path.exists(out):
                print(f"  [{idx:3d}/{total}] SKIP  {fname}")
                return None
            async with sem:
                print(f"  [{idx:3d}/{total}] → {prompt[:80]}")
                pose_cloud_name = None
                if args.pipeline == "anypose":
                    pose_path = job[5]
                    pose_cloud_name = pose_img_names.get(pose_path)
                    if not pose_cloud_name:
                        print(f"  [{idx:3d}/{total}] ✗ {fname}  (pose image not uploaded)")
                        return None
                wf = build_workflow(img_name, az, el, dist, prompt,
                                    args.seed, args.steps,
                                    args.guidance, args.lora_lightning, args.lora_angles,
                                    pipeline=args.pipeline,
                                    pose_image_filename=pose_cloud_name)
                pid = await cloud_submit(session, api_key, wf)
                pid_to_job[pid] = (idx, fname, out, prompt)
                pending.add(pid)
                return pid

        # Start submitting in background while we listen on websocket
        async def submit_all():
            submit_tasks = []
            for i, job in enumerate(jobs, 1):
                submit_tasks.append(submit_one(i, job))
            results = await asyncio.gather(*submit_tasks)
            active = [p for p in results if p is not None]
            print(f"\n  All {len(active)} jobs submitted.\n", flush=True)
            return active

        submit_task = asyncio.create_task(submit_all())

        # Listen on websocket concurrently with submission
        skipped = 0
        start = time.time()

        # Wait until at least one job is submitted or all are done
        while not pending and not submit_task.done():
            await asyncio.sleep(0.1)

        while (time.time() - start) < args.timeout:
            # Check if all done
            if submit_task.done() and not pending:
                break

            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
            except asyncio.TimeoutError:
                continue
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                msg_type = data.get("type", "")
                msg_data = data.get("data", {})
                pid = msg_data.get("prompt_id", "")

                if pid not in pending:
                    continue

                if msg_type == "executed":
                    images = msg_data.get("output", {}).get("images", [])
                    if images:
                        idx, fname, out, prompt = pid_to_job[pid]
                        try:
                            await cloud_download(session, api_key,
                                                 images[0]["filename"], out)
                            cleanup_checkerboard(out, args.image)
                            print(f"  [{idx:3d}/{total}] ✓ {fname}")
                            print(f"           prompt: {prompt}")
                            ok += 1
                        except Exception as e:
                            print(f"  [{idx:3d}/{total}] ✗ {fname}  (download: {e})")
                            fail += 1
                        pending.discard(pid)

                elif msg_type == "execution_error":
                    idx, fname, out, prompt = pid_to_job[pid]
                    err = msg_data.get("exception_message", "unknown error")
                    print(f"  [{idx:3d}/{total}] ✗ {fname}  ({err})")
                    fail += 1
                    pending.discard(pid)

            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                print("  WebSocket closed unexpectedly")
                break

        # Ensure submit task is done
        if not submit_task.done():
            await submit_task
        active_pids = submit_task.result()
        skipped = total - len(active_pids)

    # Anything still pending timed out
    for pid in pending:
        idx, fname, out, prompt = pid_to_job[pid]
        print(f"  [{idx:3d}/{total}] ✗ {fname}  (timeout)")
        fail += 1

    return ok + skipped, fail


async def _run_cloud(jobs, args):
    import aiohttp
    api_key = os.environ.get("COMFY_CLOUD_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: Set COMFY_CLOUD_API_KEY env var.\n  Get key at: https://platform.comfy.org/login")

    print(f"  Mode: CLOUD → {CLOUD_BASE}")
    print(f"  Concurrency: {args.concurrency} parallel jobs\n")

    async with aiohttp.ClientSession() as session:
        return await _process_batch_ws(jobs, args, api_key, session)


def run_cloud(jobs, args):
    return asyncio.run(_run_cloud(jobs, args))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Batch-render multi-angle poses via ComfyUI")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output", default=None, help="Output directory (default: ./multi_angle_output_{pipeline}_seed{seed})")
    p.add_argument("--cloud", action="store_true",
                   help="Use Comfy Cloud (set COMFY_CLOUD_API_KEY env var)")
    p.add_argument("--server", default="http://127.0.0.1:8188",
                   help="Local ComfyUI URL (ignored with --cloud)")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Parallel cloud jobs (Free=1, Standard=1, Creator=3, Pro=5)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--lora-angles", type=float, default=1.0)
    p.add_argument("--lora-lightning", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--azimuths", default=None, help="Subset, e.g. 0,90,180,270")
    p.add_argument("--elevations", default=None, help="Subset, e.g. -30,0,30,60")
    p.add_argument("--distances", default=None, help="Subset, e.g. 0.6,1.0,1.8")
    p.add_argument("--pipeline", default="2511",
                   choices=["2509", "2511", "anypose", "expressions", "lighting", "outfits", "poses_prompt", "angles_prompt"],
                   help="Model pipeline: 2509, 2511 (default), anypose, expressions, lighting, outfits, poses_prompt, or angles_prompt")
    p.add_argument("--pose-dir", default=None,
                   help="Directory of pose images (required for --pipeline anypose)")
    p.add_argument("--prompt-append", default="",
                   help="String to append to every generated prompt")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output is None:
        args.output = f"./multi_angle_output_{args.pipeline}_seed{args.seed}"

    # ── Parse filters ─────────────────────────────────────────────────
    azimuths = ALL_AZIMUTHS
    if args.azimuths:
        azimuths = [int(x) for x in args.azimuths.split(",")]
        for a in azimuths:
            assert a in AZIMUTH_MAP, f"Bad azimuth {a}. Valid: {list(AZIMUTH_MAP)}"

    elevations = ALL_ELEVATIONS
    if args.elevations:
        elevations = [int(x) for x in args.elevations.split(",")]
        for e in elevations:
            assert e in ELEVATION_MAP, f"Bad elevation {e}. Valid: {list(ELEVATION_MAP)}"

    distances = ALL_DISTANCES
    if args.distances:
        distances = [float(x) for x in args.distances.split(",")]
        for d in distances:
            assert d in DISTANCE_MAP, f"Bad distance {d}. Valid: {list(DISTANCE_MAP)}"

    suffix = f" {args.prompt_append}" if args.prompt_append else ""

    if args.pipeline == "angles_prompt":
        jobs = [
            (None, None, None,
             ANGLES_PROMPT_PREFIX + desc + suffix,
             f"angle_{name}.png")
            for name, desc in ANGLES_PROMPT.items()
        ]
    elif args.pipeline == "poses_prompt":
        jobs = [
            (None, None, None,
             POSES_PREFIX + desc + suffix,
             f"pose_{name}.png")
            for name, desc in POSES.items()
        ]
    elif args.pipeline == "expressions":
        jobs = [
            (None, None, None,
             EXPRESSION_PREFIX + desc + suffix,
             f"expr_{name}.png")
            for name, desc in EXPRESSIONS.items()
        ]
    elif args.pipeline == "lighting":
        jobs = [
            (None, None, None,
             LIGHTING_PREFIX + desc + suffix,
             f"light_{name}.png")
            for name, desc in LIGHTING.items()
        ]
    elif args.pipeline == "outfits":
        jobs = [
            (None, None, None,
             OUTFITS_PREFIX + desc + suffix,
             f"outfit_{name}.png")
            for name, desc in OUTFITS.items()
        ]
    elif args.pipeline == "anypose":
        # AnyPose: iterate over pose images from a directory
        if not args.pose_dir:
            sys.exit("ERROR: --pose-dir is required for --pipeline anypose")
        assert os.path.isdir(args.pose_dir), f"Pose directory not found: {args.pose_dir}"
        pose_files = sorted([
            f for f in os.listdir(args.pose_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        assert pose_files, f"No image files found in {args.pose_dir}"
        prompt = ANYPOSE_DEFAULT_PROMPT + suffix
        # jobs tuple: (az, el, dist, prompt, fname, pose_path)
        jobs = [
            (None, None, None, prompt,
             f"pose_{os.path.splitext(pf)[0]}.png",
             os.path.join(args.pose_dir, pf))
            for pf in pose_files
        ]
    else:
        prompt_fn = build_prompt_2509 if args.pipeline == "2509" else build_prompt_2511
        # For 2509, skip duplicate elevations (30° and 60° produce the same prompt)
        if args.pipeline == "2509" and not args.elevations:
            elevations = [e for e in elevations if e != 60]
        jobs = [(az, el, dist, prompt_fn(az, el, dist) + suffix, safe_filename(az, el, dist))
                for az in azimuths for el in elevations for dist in distances]

    total = len(jobs)
    mode = "CLOUD" if args.cloud else "LOCAL"

    print(f"\n{'='*64}")
    if args.pipeline == "anypose":
        print(f"  Qwen Image Edit — AnyPose Batch Renderer")
    elif args.pipeline == "angles_prompt":
        print(f"  Qwen Image Edit — Prompt Angle Batch Renderer")
    elif args.pipeline == "poses_prompt":
        print(f"  Qwen Image Edit — Prompt Pose Batch Renderer")
    elif args.pipeline == "expressions":
        print(f"  Qwen Image Edit — Expression Batch Renderer")
    elif args.pipeline == "lighting":
        print(f"  Qwen Image Edit — Lighting Variation Renderer")
    elif args.pipeline == "outfits":
        print(f"  Qwen Image Edit — Outfit Variation Renderer")
    else:
        print(f"  Qwen Image Edit {args.pipeline} — Multi-Angle Batch Renderer")
    print(f"{'='*64}")
    print(f"  Input   : {args.image}")
    if args.pipeline == "anypose":
        print(f"  Pose Dir: {args.pose_dir}  ({total} poses)")
    elif args.pipeline == "angles_prompt":
        print(f"  Angles: {total}")
    elif args.pipeline == "poses_prompt":
        print(f"  Poses: {total}")
    elif args.pipeline == "expressions":
        print(f"  Expressions: {total}")
    elif args.pipeline == "lighting":
        print(f"  Lighting: {total} variations")
    elif args.pipeline == "outfits":
        print(f"  Outfits: {total} variations")
    print(f"  Output  : {args.output}")
    if args.pipeline not in ("anypose", "expressions", "lighting", "outfits", "poses_prompt", "angles_prompt"):
        print(f"  Poses   : {total}  ({len(azimuths)} az × {len(elevations)} el × {len(distances)} dist)")
    print(f"  Steps   : {args.steps}  |  CFG: {args.guidance}  |  Seed: {args.seed}")
    print(f"  Pipeline: {args.pipeline}")
    print(f"  Target  : {mode}" + (f"  (concurrency: {args.concurrency})" if args.cloud else ""))
    print(f"{'='*64}\n")

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            prompt = job[3]
            fname = job[4]
            print(f"  [{i:3d}/{total}] {prompt[:80]}")
            print(f"           → {fname}\n")
        print(f"Total: {total} images. Remove --dry-run to execute.")
        return

    assert os.path.isfile(args.image), f"Image not found: {args.image}"
    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    ok, fail = (run_cloud if args.cloud else run_local)(jobs, args)
    elapsed = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  COMPLETE  —  {ok}/{total} rendered, {fail} failed, {elapsed:.0f}s total")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
