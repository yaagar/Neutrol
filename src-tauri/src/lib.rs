// lib.rs — Neutrol Tauri application logic.
//
// Launches the neutrol-backend sidecar (PyInstaller-packaged FastAPI server)
// when the app starts, and kills it cleanly when the last window closes.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::{process::CommandChild, ShellExt};

// Holds the backend process handle so we can kill it on app exit.
struct SidecarChild(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Spawn the packaged Python backend. "neutrol-backend" must match
            // the externalBin name in tauri.conf.json (without the platform suffix).
            let sidecar_cmd = app.shell().sidecar("neutrol-backend")?;
            let (_rx, child) = sidecar_cmd.spawn()?;
            app.manage(SidecarChild(Mutex::new(Some(child))));
            Ok(())
        })
        .on_window_event(|window, event| {
            // Kill the backend when the window is destroyed (app closed).
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<SidecarChild>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.take() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Neutrol");
}
