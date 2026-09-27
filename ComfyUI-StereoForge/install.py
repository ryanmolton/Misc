"""
Installs the two packages whose own dependency pins would downgrade ComfyUI's
environment (numpy<2, old pillow, xformers, open3d ...).  ComfyUI-Manager runs this
automatically; otherwise run it once with ComfyUI's python:  python install.py
"""
import os
import subprocess
import sys

PKGS = [
    "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git",
    "simple-lama-inpainting",
]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(here, "requirements.txt")])
for p in PKGS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", p])
