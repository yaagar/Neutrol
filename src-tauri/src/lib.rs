// lib.rs — Neutrol Tauri application logic.
//
// Responsibilities:
//   1. Launch the neutrol-backend sidecar (PyInstaller-packaged FastAPI server)
//      as soon as the app starts. The backend binds to 127.0.0.1:8000.
//   2. Kill the sidecar process cleanly when the last window is destroyed,
//      so the backend never lingers after the user closes the app.
//
// The React frontend communicates with the backend over HTTP/WebSocket at
// 127.0.0.1:8000, exactly as in development mode.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::{process::CommandChild, ShellExt};

// SidecarChild wraps the backend process handle in a Mutex so it can be
// stored in Tauri's managed state and safely accessed from multiple threads.
struct SidecarChild(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Register the shell plugin — needed to spawn sidecar processes.
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Spawn the Python backend sidecar.
            // "neutrol-backend" matches the `externalBin` name in tauri.conf.json.
            // The sidecar binary must already be present in src-tauri/binaries/
            // before `tauri build` is run (see build_backend.py).
            let sidecar_cmd = app.shell().sidecar("neutrol-backend")?;
            let (_rx, child) = sidecar_cmd.spawn()?;

            // Store the child handle in managed state so we can kill it later.
            app.manage(SidecarChild(Mutex::new(Some(child))));
            Ok(())
        })
        // Kill the backend sidecar when the last window closes.
        // WindowEvent::Destroyed fires once per window; on single-window apps
        // this is equivalent to the app exiting.
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<SidecarChild>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.take() {
                            // Best-effort kill — ignore errors (process may
                            // have already exited if the user killed it).
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Neutrol");
}
