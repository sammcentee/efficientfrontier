@echo off
setlocal
title Portfolio Lab
pushd "%~dp0" || exit /b 1
where cargo >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where cargo >nul 2>nul
if errorlevel 1 (
  echo Portfolio Lab needs the Rust toolchain.
  echo Install it from https://rustup.rs, then double-click this launcher again.
  set "portfolio_exit=1"
  goto finish
)
echo Portfolio Lab - starting the desktop app
echo The first start compiles the app. It needs internet and can take a few minutes.
cargo run --release --locked -p portfolio-lab -- %*
set "portfolio_exit=%errorlevel%"
:finish
if not "%portfolio_exit%"=="0" (
  echo.
  echo Other help is in README.md.
  pause
)
popd
exit /b %portfolio_exit%
