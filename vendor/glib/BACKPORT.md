# GLib 0.18.5 security backport

Source: the complete crates.io `glib` 0.18.5 archive,
<https://static.crates.io/crates/glib/glib-0.18.5.crate>.

Archive SHA-256, verified against the original `Cargo.lock`:
`233daaf6e83ae6a12a52055f568f9d7cf4671dabb78ff9560ab6da230ce00ee5`.

This directory preserves all upstream files, with `LICENSE` and `COPYRIGHT`.
The only source changes are the two lines from
[gtk-rs/gtk-rs-core PR #1343](https://github.com/gtk-rs/gtk-rs-core/pull/1343/files).
In `src/variant_iter.rs`, `VariantStrIter::impl_get` declares `p` as mutable and
passes `&mut p` to `g_variant_get_child`, instead of `&p`.
This corrects the write through an immutable reference before `CStr::from_ptr`.
This note is a separate local addition.

Advisory: [RUSTSEC-2024-0429](https://rustsec.org/advisories/RUSTSEC-2024-0429.html),
also [GHSA-wrw7-89jp-8q8g](https://github.com/advisories/GHSA-wrw7-89jp-8q8g).
The upstream advisory lists 0.20.0 as the first fixed release.
The Linux Tauri/GTK dependency graph requires GLib 0.18.
The root `[patch.crates-io]` selects this backport without an API change or a
dependency upgrade. The crate retains version 0.18.5, so tools that check only
the version can still report the advisory.

The Linux desktop unit tests cover forward, backward, and mixed string iteration,
with empty and Unicode strings. Run the optimized regression from the root:

```sh
cargo test --locked --release -p portfolio-lab --bin portfolio-lab tests::glib_variant_str_iter
```

Keep this override until the Linux dependency graph can use an upstream fixed
release. Do not remove it based only on the absence of a direct app caller.
