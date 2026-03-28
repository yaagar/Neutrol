#!/usr/bin/env bash
# package_app.sh — One-command build for Neutrol on Linux / macOS.
#
# Steps:
#   1. Build the Python backend into a single binary with PyInstaller.
#   2. Install frontend npm dependencies (if not already installed).
#   3. Run `tauri build` which: builds the React app, then packages
#      everything (Rust app + frontend + sidecar binary) into an installer.
#
# Requirements:
#   - Python 3.11+ with pip (in PATH or activate your venv first)
#   - pip install pyinstaller  (in the same env as the backend deps)
#   - Node.js + npm (in PATH)
#   - Rust + cargo (in PATH)  — install from https://rustup.rs
#   - Linux: libwebkit2gtk-4.1-dev, libgtk-3-dev, libayatana-appindicator3-dev
#
# Output:
#   src-tauri/target/release/bundle/  — contains the .deb / .AppImage / .dmg

set -euo pipefail

echo "============================================================"
echo " Neutrol — Desktop App Builder (Linux / macOS)"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Build the Python backend sidecar
echo "[1/3] Building Python backend with PyInstaller..."
python3 build_backend.py
echo ""

# Step 2: Install frontend dependencies
echo "[2/3] Installing frontend dependencies..."
cd frontend
npm install
cd ..
echo ""

# Step 3: Build the Tauri app
echo "[3/3] Building Tauri desktop app..."
cd frontend
npm run tauri build
cd ..

echo ""
echo "============================================================"
echo " Build complete!"
echo " Installer: src-tauri/target/release/bundle/"
echo "============================================================"
