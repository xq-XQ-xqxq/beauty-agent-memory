$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$streamlitExe = Join-Path $projectDir ".venv\Scripts\streamlit.exe"
$appFile = Join-Path $projectDir "streamlit_app.py"
$demoUrl = "http://localhost:8501"

if (-not (Test-Path -LiteralPath $streamlitExe)) {
    Write-Host "[错误] 未找到 Streamlit 运行环境：$streamlitExe" -ForegroundColor Red
    Write-Host "请先在项目目录运行：.\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Read-Host "按回车键退出"
    exit 1
}

$dashscopeKey = $env:DASHSCOPE_API_KEY
if ([string]::IsNullOrWhiteSpace($dashscopeKey)) {
    $dashscopeKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
}
if ([string]::IsNullOrWhiteSpace($dashscopeKey)) {
    $dashscopeKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "Machine")
}
if ([string]::IsNullOrWhiteSpace($dashscopeKey)) {
    Write-Host "[错误] 未找到 DASHSCOPE_API_KEY（当前终端、用户级或机器级）。" -ForegroundColor Red
    Write-Host '可执行：[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "你的Key", "User")'
    Read-Host "按回车键退出"
    exit 1
}
$env:DASHSCOPE_API_KEY = $dashscopeKey

$env:MEM0_COLLECTION_NAME = "customer_workbench_demo_v1_1024"
$env:MEM0_DIR = Join-Path $projectDir "data\mem0_home"

$existingListener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    Write-Host "工作台已经在运行，正在打开页面……" -ForegroundColor Green
    Start-Process $demoUrl
    exit 0
}

Write-Host "正在启动共情记忆工作台……" -ForegroundColor Cyan
Write-Host "关闭此窗口即可停止工作台。"
Start-Process $demoUrl
& $streamlitExe run $appFile --server.port 8501
