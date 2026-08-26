param(
    [string]$Version = "1.6.12",
    [switch]$CompileOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$packageName = "本地多格式全文搜索工具"
$source = Join-Path (Join-Path $repo "dist") ("$packageName-$Version")
$iss = Join-Path $repo "installer\LocalFullTextSearch.iss"
$isccCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)
$iscc = $isccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "未找到 Inno Setup 6 ISCC.exe，请先安装 Inno Setup。"
}
if (-not (Test-Path -LiteralPath (Join-Path $source "$packageName.exe"))) {
    throw "未找到 PyInstaller 发布目录：$source"
}

if (-not $CompileOnly) {
    Write-Host "[setup] 编译安装程序..."
    & $iscc "/DMyAppVersion=$Version" "/DSourceDir=$source" $iss
    if ($LASTEXITCODE -ne 0) { throw "ISCC 编译失败：$LASTEXITCODE" }
}

$setup = Join-Path (Join-Path $repo "dist") ("$packageName-安装程序-$Version.exe")
if (-not (Test-Path -LiteralPath $setup)) { throw "安装程序未生成：$setup" }
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash
$size = (Get-Item -LiteralPath $setup).Length
Write-Host "SETUP_OK size=$size sha256=$hash path=$setup"
