fn main() {
    // Reads tauri.conf.json, and on Windows puts the icon and the manifest into the .exe.
    tauri_build::build();
}
