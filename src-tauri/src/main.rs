// main.rs — Tauri application entry point.
//
// Tauri requires this file to exist. All real logic lives in lib.rs so that
// the library crate can be tested independently.
//
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// suppresses the terminal window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    neutrol_lib::run();
}
