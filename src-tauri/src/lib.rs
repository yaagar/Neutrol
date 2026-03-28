// lib.rs — Neutrol Tauri application logic.
//
// Dev mode: the FastAPI backend runs as a separate process (Terminal 1).
// Production mode: the backend sidecar is launched automatically here.
// To enable the sidecar for a production build, see build_backend.py and
// the commented-out sidecar code below.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .run(tauri::generate_context!())
        .expect("error while running Neutrol");
}
