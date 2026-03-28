"""
build_backend.py — Package the Neutrol FastAPI backend with PyInstaller.

What this script does:
  1. Calls PyInstaller to bundle backend/main.py and all its dependencies
     into a single executable file named 'neutrol-backend'.
  2. Detects the current platform target triple (e.g. x86_64-pc-windows-msvc).
  3. Copies the output binary to src-tauri/binaries/ with the correct name
     that Tauri expects for sidecar registration.

Why PyInstaller?
  PyInstaller bundles the Python interpreter, all pip packages, and the app
  code into a self-contained executable. The user does NOT need Python installed.
  Tauri then ships this executable inside the installer alongside the Rust app.

Usage:
  python build_backend.py

Requirements:
  pip install pyinstaller

The output is placed at:
  src-tauri/binaries/neutrol-backend-<target_triple>[.exe]

Run this before `npm run tauri build` (or the package_app script does it for you).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def get_target_triple() -> str:
    """
    Return the Rust-style target triple for the current OS and architecture.

    Tauri requires sidecar binaries to be named:
      <name>-<target_triple>[.exe]

    This matches what `rustc -vV` outputs for `host`.
    """
    machine = platform.machine().lower()
    system  = platform.system().lower()

    # Normalize architecture names
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        arch = machine

    if system == "windows":
        return f"{arch}-pc-windows-msvc"
    elif system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "linux":
        return f"{arch}-unknown-linux-gnu"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_main = os.path.join(project_root, "backend", "main.py")
    dist_dir     = os.path.join(project_root, "dist")
    binaries_dir = os.path.join(project_root, "src-tauri", "binaries")

    os.makedirs(binaries_dir, exist_ok=True)

    target_triple = get_target_triple()
    is_windows    = platform.system().lower() == "windows"
    exe_suffix    = ".exe" if is_windows else ""

    # PyInstaller output binary (before renaming)
    built_exe = os.path.join(dist_dir, f"neutrol-backend{exe_suffix}")
    # Final destination Tauri expects
    dest_name = f"neutrol-backend-{target_triple}{exe_suffix}"
    dest_path = os.path.join(binaries_dir, dest_name)

    print(f"[build_backend] Target triple : {target_triple}")
    print(f"[build_backend] Backend entry : {backend_main}")
    print(f"[build_backend] Output        : {dest_path}")
    print()

    # ----------------------------------------------------------
    # Run PyInstaller
    # ----------------------------------------------------------
    # Flags:
    #   --onefile        Bundle everything into a single executable
    #   --name           Output binary name (without platform suffix)
    #   --distpath       Where to put the output (project/dist/)
    #   --workpath       Temp build files (project/build/)
    #   --specpath       Where to put the .spec file
    #   --noconfirm      Overwrite without asking
    #   --log-level WARN Suppress noisy INFO output
    #
    # Hidden imports: some libraries that PyInstaller misses with static analysis.
    hidden = [
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "anyio",
        "anyio._backends._asyncio",
        "scipy.special.cython_special",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name",       "neutrol-backend",
        "--distpath",   dist_dir,
        "--workpath",   os.path.join(project_root, "build"),
        "--specpath",   project_root,
        "--noconfirm",
        "--log-level",  "WARN",
    ]
    for h in hidden:
        cmd += ["--hidden-import", h]
    cmd.append(backend_main)

    print("[build_backend] Running PyInstaller…")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        print("[build_backend] ERROR: PyInstaller failed.", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------
    # Copy to src-tauri/binaries/ with the target-triple suffix
    # ----------------------------------------------------------
    if not os.path.exists(built_exe):
        print(f"[build_backend] ERROR: Expected output not found: {built_exe}", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(built_exe, dest_path)
    print(f"[build_backend] Copied to: {dest_path}")
    print("[build_backend] Done. Run 'npm run tauri build' (or package_app) next.")


if __name__ == "__main__":
    main()
