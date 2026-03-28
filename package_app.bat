@echo off
:: package_app.bat — One-command build for Neutrol on Windows.
::
:: Steps:
::   1. Build the Python backend into a single .exe with PyInstaller.
::   2. Install frontend npm dependencies (if not already installed).
::   3. Run `tauri build` which: builds the React app, then packages
::      everything (Rust app + frontend + sidecar .exe) into an installer.
::
:: Requirements:
::   - Python 3.11+ with pip (in PATH or a venv activated)
::   - pip install pyinstaller  (in the same env as the backend deps)
::   - Node.js + npm (in PATH)
::   - Rust + cargo (in PATH)  — install from https://rustup.rs
::   - Visual Studio Build Tools (for Rust on Windows)
::
:: Output:
::   src-tauri/target/release/bundle/  — contains the .msi / .exe installer

setlocal enabledelayedexpansion

echo ============================================================
echo  Neutrol — Desktop App Builder (Windows)
echo ============================================================
echo.

:: Step 1: Build the Python backend sidecar
echo [1/3] Building Python backend with PyInstaller...
python build_backend.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Backend build failed. See output above.
    pause
    exit /b 1
)
echo.

:: Step 2: Install frontend dependencies
echo [2/3] Installing frontend dependencies...
cd frontend
call npm install
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed.
    pause
    exit /b 1
)
echo.

:: Step 3: Build the Tauri app (builds frontend + packages everything)
echo [3/3] Building Tauri desktop app...
call npm run tauri build
if %ERRORLEVEL% neq 0 (
    echo ERROR: Tauri build failed. See output above.
    pause
    exit /b 1
)
cd ..

echo.
echo ============================================================
echo  Build complete!
echo  Installer: src-tauri\target\release\bundle\
echo ============================================================
pause
