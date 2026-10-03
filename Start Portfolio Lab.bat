@echo off
setlocal
title Portfolio Lab
pushd "%~dp0" || exit /b 1
where pymanager >nul 2>nul
if errorlevel 1 (
  py -3.14 launch.py %*
) else (
  pymanager exec -V:3.14 launch.py %*
)
set "portfolio_exit=%errorlevel%"
if not "%portfolio_exit%"=="0" (
  echo.
  echo If Python was not found, install the Python Install Manager from:
  echo https://www.python.org/downloads/windows/
  echo Then double-click this launcher again. Other help is in README.md.
  pause
)
popd
exit /b %portfolio_exit%
