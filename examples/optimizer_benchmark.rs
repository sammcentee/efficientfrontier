//! Bounded synthetic optimizer benchmark; results are not investment evidence.
//!
//! Run: `cargo run --release --example optimizer_benchmark -- --sizes 100 500 1000 --output results/benchmarks.json`
//!
//! Each case runs in a fresh worker process with a time budget. The budget
//! covers data generation, optimization and checks; `optimizer_seconds`
//! measures only `optimize()`. Peak RSS is the whole worker process (Linux only).

use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use efficient_frontier::core::optimize;
use efficient_frontier::frame::{LabeledMatrix, Series};
use nalgebra::DMatrix;
use serde_json::{Value, json};

struct Args {
    sizes: Vec<usize>,
    frontier_points: usize,
    max_weight: f64,
    timeout: f64,
    output: Option<String>,
    worker: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        sizes: vec![100, 500, 1000],
        frontier_points: 40,
        max_weight: 1.0,
        timeout: 60.0,
        output: None,
        worker: false,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let value = |i: usize| raw.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", raw[i]));
        match raw[i].as_str() {
            "--sizes" => {
                args.sizes.clear();
                while i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                    i += 1;
                    args.sizes.push(raw[i].parse().map_err(|_| "sizes must be positive integers")?);
                }
            }
            "--frontier-points" => {
                args.frontier_points = value(i)?.parse().map_err(|_| "frontier-points must be an integer")?;
                i += 1;
            }
            "--max-weight" => {
                args.max_weight = value(i)?.parse().map_err(|_| "max-weight must be a number")?;
                i += 1;
            }
            "--timeout" => {
                args.timeout = value(i)?.parse().map_err(|_| "timeout must be a number")?;
                i += 1;
            }
            "--output" => {
                args.output = Some(value(i)?);
                i += 1;
            }
            "--worker" => args.worker = true,
            other => return Err(format!("unknown argument {other}")),
        }
        i += 1;
    }
    if args.sizes.is_empty() || args.sizes.contains(&0) {
        return Err("sizes must be positive integers".into());
    }
    if args.frontier_points < 2 || !(args.max_weight > 0.0 && args.max_weight <= 1.0) {
        return Err("frontier-points must be at least 2 and max-weight at most 1".into());
    }
    if args.sizes.iter().any(|size| (*size as f64) * args.max_weight < 1.0 - 1e-12) {
        return Err("every size * max-weight must be at least 1".into());
    }
    if !args.timeout.is_finite() || args.timeout <= 0.0 {
        return Err("timeout must be finite and positive".into());
    }
    Ok(args)
}

/// SplitMix64 with Box-Muller normals: reproducible without extra dependencies.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(f64::MIN_POSITIVE), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}

fn run_case(args: &Args) -> Result<Value, String> {
    let count = args.sizes[0];
    let mut random = Random(20261003 + count as u64);
    let factors = count.min(24);
    let loadings = DMatrix::from_fn(count, factors, |_, _| random.normal() * 0.05 / (factors as f64).sqrt());
    let mut sigma = &loadings * loadings.transpose();
    for i in 0..count {
        sigma[(i, i)] += (0.01 + 0.03 * random.next()).powi(2);
    }
    let labels: Vec<String> = (0..count).map(|i| format!("SYNTHETIC_{i:05}")).collect();
    let means: Vec<f64> = (0..count).map(|i| 0.03 + 0.17 * i as f64 / (count.max(2) - 1) as f64).collect();
    let mu = Series::new(labels.clone(), means.clone());
    let covariance = LabeledMatrix::square(labels, sigma);
    let start = Instant::now();
    let result =
        optimize(&mu, &covariance, 0.02, args.max_weight, args.frontier_points).map_err(|error| error.to_string())?;
    let elapsed = start.elapsed().as_secs_f64();
    let tolerance = 1e-6;
    let rows: Vec<&Vec<f64>> =
        result.portfolios.values().map(|p| &p.weights.values).chain(result.frontier.weights.iter()).collect();
    let checks = json!({
        "finite_weights": rows.iter().all(|w| w.iter().all(|v| v.is_finite())),
        "sum_to_one": rows.iter().all(|w| (w.iter().sum::<f64>() - 1.0).abs() <= tolerance),
        "nonnegative": rows.iter().all(|w| w.iter().all(|v| *v >= -tolerance)),
        "position_cap": rows.iter().all(|w| w.iter().all(|v| *v <= args.max_weight + tolerance)),
        "frontier_returns": result.frontier.weights.iter().zip(&result.frontier.expected_return)
            .all(|(w, expected)| (w.iter().zip(&means).map(|(a, b)| a * b).sum::<f64>() - expected).abs() <= tolerance),
    });
    if checks.as_object().unwrap().values().any(|value| value != true) {
        return Err(format!("Invalid optimizer output: {checks}"));
    }
    Ok(json!({
        "status": "ok", "optimizer_seconds": elapsed, "peak_rss_mib": peak_rss_mib(),
        "actual_frontier_points": result.frontier.len(), "checks": checks, "warnings": result.warnings,
        "minimum_variance": result.portfolios["Minimum volatility"].volatility.powi(2),
        "maximum_sharpe": result.portfolios.get("Maximum Sharpe").map(|p| p.sharpe),
    }))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("optimizer_benchmark: {message}");
            return ExitCode::from(2);
        }
    };
    if args.worker {
        return match run_case(&args) {
            Ok(case) => {
                println!("{case}");
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        };
    }
    let mut report = json!({
        "model": "Synthetic annual means and dense, full-rank covariance from up to 24 factors plus positive diagonal; seed 20261003 + assets. No market data or investment claims.",
        "metadata": {
            "efficient_frontier": env!("CARGO_PKG_VERSION"),
            "target": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            "cpu_count": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "timeout_seconds": args.timeout,
        },
        "cases": [],
    });
    let executable = std::env::current_exe().expect("benchmark executable");
    let mut failed = false;
    for size in &args.sizes {
        let mut case = json!({"implementation": "current", "assets": size,
                              "requested_frontier_points": args.frontier_points, "max_weight": args.max_weight});
        let start = Instant::now();
        let child = Command::new(&executable)
            .args(["--worker", "--sizes", &size.to_string(), "--frontier-points", &args.frontier_points.to_string()])
            .args(["--max-weight", &args.max_weight.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let outcome = child.map_err(|error| error.to_string()).and_then(|mut child| {
            let deadline = Instant::now() + Duration::from_secs_f64(args.timeout);
            loop {
                if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                    return child.wait_with_output().map(Some).map_err(|e| e.to_string());
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        let object = case.as_object_mut().unwrap();
        match outcome {
            Ok(Some(output)) if output.status.success() => {
                let result: Value = serde_json::from_slice(&output.stdout).unwrap_or(json!({"status": "error"}));
                object.extend(result.as_object().cloned().unwrap_or_default());
            }
            Ok(Some(output)) => {
                object.insert("status".into(), json!("error"));
                object.insert("stderr".into(), json!(String::from_utf8_lossy(&output.stderr)));
            }
            Ok(None) => {
                object.insert("status".into(), json!("timeout"));
            }
            Err(message) => {
                object.insert("status".into(), json!("error"));
                object.insert("stderr".into(), json!(message));
            }
        }
        object.insert("process_seconds".into(), json!(start.elapsed().as_secs_f64()));
        failed |= object["status"] != "ok";
        println!("{case}");
        report["cases"].as_array_mut().unwrap().push(case);
        if let Some(path) = &args.output {
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, serde_json::to_string_pretty(&report).unwrap() + "\n").expect("write the report");
        }
    }
    if args.output.is_none() {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
