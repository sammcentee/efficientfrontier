//! The app's window, driven through WebDriver from the start page to a saved report.
//!
//! On Linux, `tauri-driver` starts WebKitWebDriver, which opens the window. On
//! Windows, the test starts the app with WebView2's remote debugging port, and
//! Microsoft Edge Driver attaches to it, as Microsoft documents for WebView2
//! apps. The test is ignored by default, because it needs a release build,
//! these drivers and a screen. On Linux, give it a virtual screen, not the desktop. Without
//! `GDK_BACKEND=x11`, GTK can still open the window on a Wayland desktop, such
//! as WSLg:
//!
//! ```sh
//! cargo build --release -p portfolio-lab
//! env -u WAYLAND_DISPLAY GDK_BACKEND=x11 xvfb-run -a cargo test -p portfolio-lab --test window -- --ignored
//! ```
//!
//! On Linux the test also saves files through GTK's Save dialog with `xdotool`.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const DRIVER: &str = "http://127.0.0.1:4444";
/// The WebDriver key codes of Control and Enter.
const CONTROL: &str = "\u{E009}";
const ENTER: &str = "\u{E007}";
/// The WebDriver name of an element reference.
const ELEMENT: &str = "element-6066-11e4-a52e-4f735466cecf";
/// The longest wait for one step, such as a build of the demo study.
const STEP: Duration = Duration::from_secs(120);
/// WebView2's remote debugging port for Microsoft Edge Driver on Windows.
const DEBUG_PORT: u16 = 9222;

#[test]
#[ignore = "opens the app's window: needs a release build, a WebDriver and a screen"]
fn the_window_builds_studies_and_saves_files() {
    let app = app();
    let files = tempfile::tempdir().expect("a temporary folder");
    let _driver = Driver::start(&app, &files.path().join("webview2"));
    let window = Window::open(&app);

    window.wait("the start page", "return ['landing', 'error'].includes(document.body.dataset.page)");
    let title = window.script("return document.querySelector('h1').textContent");
    assert!(
        title.as_str().is_some_and(|title| title.starts_with("Efficient Frontier Analysis for US Equities")),
        "{title}"
    );
    window.wait("typeset equations", "return document.querySelectorAll('.katex').length >= 3");

    build(&window, "Demo", None);
    window.wait("three charts", "return document.querySelectorAll('.js-plotly-plot').length >= 3");

    window.press(&["4"]);
    window.wait("the Data & method view", "return document.body.dataset.page === 'method'");
    if cfg!(target_os = "linux") {
        let prices = files.path().join("prices.csv");
        window.script("document.querySelector('#download_prices').click()");
        save_in_dialog(&prices);
        let text = std::fs::read_to_string(&prices).expect("the saved prices");
        assert!(text.starts_with("Date,"), "{}", &text[..text.len().min(80)]);
    }
    window.press(&["1"]);
    window.wait("the Portfolio view", "return document.body.dataset.page === 'overview'");
    if cfg!(target_os = "linux") {
        let report = files.path().join("report.zip");
        window.press(&["e"]);
        save_in_dialog(&report);
        let archive = zip::ZipArchive::new(std::fs::File::open(&report).expect("the saved report")).expect("a zip");
        let names: Vec<String> = archive.file_names().filter_map(Result::ok).map(|name| name.into_owned()).collect();
        assert!(names.len() >= 20 && names.iter().any(|name| name == "report.html"), "{names:?}");
    }

    // The upload goes through the app's protocol, so this checks the request body.
    // The drawer keeps the demo study's 20% holding limit, which six stocks need.
    let prices = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/demo/demo_prices.csv");
    build(&window, "Upload CSV", Some(&prices));

    window.close();
}

/// Open the settings, choose a market and a prices file, and build with Ctrl+Enter.
fn build(window: &Window, market: &str, prices: Option<&Path>) {
    window.press(&["s"]);
    window.wait("the settings drawer", "return document.querySelector('#setup_form [data-build]') !== null");
    window.script(&format!("document.querySelector(\"#setup_form input[name='market'][value='{market}']\").click()"));
    if let Some(path) = prices {
        window.choose_file("#prices_upload", path);
    }
    // A new page has no marker, so the wait below cannot see the page before the build.
    window.script("window.plBeforeBuild = true");
    window.press(&[CONTROL, ENTER]);
    window.wait(
        &format!("the {market} study"),
        "return window.plBeforeBuild === undefined && document.body.dataset.ready === '1' && document.body.dataset.page === 'overview'",
    );
    let errors = window.script("return [...document.querySelectorAll('.pl-alert-error')].map(e => e.textContent)");
    assert_eq!(errors, json!([]), "the {market} study shows errors");
}

/// The release build of the app.
fn app() -> PathBuf {
    let name = if cfg!(windows) { "portfolio-lab.exe" } else { "portfolio-lab" };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/release").join(name);
    assert!(path.exists(), "build the app first: cargo build --release -p portfolio-lab");
    path
}

/// Wait until `ready` is true, or fail after `limit`.
fn wait_until(what: &str, limit: Duration, mut ready: impl FnMut() -> bool) {
    let start = Instant::now();
    while !ready() {
        assert!(start.elapsed() < limit, "no {what} after {} seconds", limit.as_secs());
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// The WebDriver server, and on Windows the app, stopped when the test ends.
struct Driver(Vec<Child>);

impl Driver {
    /// On Windows, `profile` is a new WebView2 profile, so the test leaves the app's own profile alone.
    fn start(app: &Path, profile: &Path) -> Self {
        let children = if cfg!(windows) {
            let app = Command::new(app)
                .env("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", format!("--remote-debugging-port={DEBUG_PORT}"))
                .env("WEBVIEW2_USER_DATA_FOLDER", profile)
                .spawn()
                .expect("the app");
            let driver = Command::new("msedgedriver")
                .arg("--port=4444")
                .stdout(Stdio::null())
                .spawn()
                .expect("Microsoft Edge Driver (msedgedriver) on PATH");
            vec![app, driver]
        } else {
            let driver = Command::new("tauri-driver")
                .stdout(Stdio::null())
                .spawn()
                .expect("tauri-driver on PATH: cargo install tauri-driver --locked");
            vec![driver]
        };
        let driver = Driver(children);
        wait_until("the WebDriver server", Duration::from_secs(30), || command("GET", "/status", None).is_ok());
        driver
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        for child in self.0.iter_mut().rev() {
            // SIGTERM lets tauri-driver stop WebKitWebDriver.
            if cfg!(unix) {
                let _ = Command::new("kill").arg("-TERM").arg(child.id().to_string()).status();
            } else {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

/// One WebDriver command. Returns the `value` of the response, or the error.
fn command(method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
    let agent: ureq::Agent =
        ureq::Agent::config_builder().timeout_global(Some(STEP)).http_status_as_error(false).build().into();
    let url = format!("{DRIVER}{path}");
    let response = match method {
        "POST" => agent.post(&url).send_json(body.unwrap_or_else(|| json!({}))),
        "DELETE" => agent.delete(&url).call(),
        _ => agent.get(&url).call(),
    };
    let mut response = response.map_err(|error| error.to_string())?;
    let status = response.status();
    let value: Value = response.body_mut().read_json().map_err(|error| error.to_string())?;
    if status.is_success() { Ok(value["value"].clone()) } else { Err(format!("{status}: {}", value["value"])) }
}

/// A WebDriver session: the app's window.
struct Window {
    session: String,
}

impl Window {
    fn open(app: &Path) -> Self {
        let options = if cfg!(windows) {
            let version = format!("http://127.0.0.1:{DEBUG_PORT}/json/version");
            wait_until("the app's WebView2", STEP, || ureq::get(&version).call().is_ok());
            json!({ "browserName": "webview2", "ms:edgeOptions": { "debuggerAddress": format!("127.0.0.1:{DEBUG_PORT}") } })
        } else {
            json!({ "tauri:options": { "application": app } })
        };
        let capabilities = json!({ "capabilities": { "alwaysMatch": options } });
        let value = command("POST", "/session", Some(capabilities)).expect("a WebDriver session");
        Window { session: value["sessionId"].as_str().expect("a session id").to_string() }
    }

    fn try_command(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        command(method, &format!("/session/{}{path}", self.session), body)
    }

    fn command(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        self.try_command(method, path, body).unwrap_or_else(|error| panic!("{method} {path}: {error}"))
    }

    /// Run a script in the page and return its result.
    fn script(&self, script: &str) -> Value {
        self.command("POST", "/execute/sync", Some(json!({ "script": script, "args": [] })))
    }

    /// Wait until a script returns true. A page that loads meanwhile counts as false.
    fn wait(&self, what: &str, script: &str) {
        let body = json!({ "script": script, "args": [] });
        wait_until(what, STEP, || {
            self.try_command("POST", "/execute/sync", Some(body.clone())) == Ok(Value::Bool(true))
        });
    }

    /// Press keys together: each key goes down in order, then up in reverse order.
    ///
    /// WebKitWebDriver refuses key input while a page loads or a download is open, so the press waits for both.
    fn press(&self, keys: &[&str]) {
        self.wait(
            "a loaded page",
            "return document.readyState === 'complete' && document.querySelector('[data-busy]') === null",
        );
        let mut actions: Vec<Value> = keys.iter().map(|key| json!({ "type": "keyDown", "value": key })).collect();
        actions.extend(keys.iter().rev().map(|key| json!({ "type": "keyUp", "value": key })));
        let body = json!({ "actions": [{ "type": "key", "id": "keyboard", "actions": actions }] });
        self.command("POST", "/actions", Some(body));
    }

    /// Choose a file in a file input, as a person does in the file dialog.
    fn choose_file(&self, css: &str, path: &Path) {
        let found = self.command("POST", "/element", Some(json!({ "using": "css selector", "value": css })));
        let element = found[ELEMENT].as_str().expect("an element").to_string();
        let path = path.canonicalize().expect("an existing file");
        self.command("POST", &format!("/element/{element}/value"), Some(json!({ "text": path })));
    }

    /// End the session, which closes the window.
    fn close(self) {
        self.command("DELETE", "", None);
    }
}

/// Type a path into GTK's Save dialog and press Enter, then wait for the file.
///
/// A virtual screen has no window manager, so the test gives the keyboard to
/// the dialog, and afterwards back to the app's window.
fn save_in_dialog(path: &Path) {
    let dialog = visible_window("^Save File$");
    for arguments in [
        vec!["windowfocus", dialog.as_str()],
        vec!["key", "--clearmodifiers", "ctrl+a"],
        vec!["type", "--delay", "5", path.to_str().expect("a UTF-8 path")],
        vec!["key", "Return"],
    ] {
        std::thread::sleep(Duration::from_millis(300));
        xdotool(&arguments);
    }
    wait_until("the saved file", STEP, || path.metadata().is_ok_and(|metadata| metadata.len() > 0));
    xdotool(&["windowfocus", visible_window("^Portfolio Lab$").as_str()]);
}

/// The X window identifier of a visible window with a title that matches `pattern`.
fn visible_window(pattern: &str) -> String {
    let found = Command::new("timeout")
        .args(["60", "xdotool", "search", "--sync", "--onlyvisible", "--name", pattern])
        .output()
        .expect("xdotool");
    String::from_utf8_lossy(&found.stdout).lines().last().unwrap_or_else(|| panic!("no window {pattern}")).to_string()
}

fn xdotool(arguments: &[&str]) {
    let status = Command::new("xdotool").args(arguments).status().expect("xdotool");
    assert!(status.success(), "xdotool {arguments:?}");
}
