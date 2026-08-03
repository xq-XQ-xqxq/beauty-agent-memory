@echo off
chcp 65001 >nul
title 共情记忆工作台
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
if errorlevel 1 pause
