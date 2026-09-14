param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PytestArgs
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "项目虚拟环境不存在：$python"
}

# Never attach Qt, multiprocessing or long-running SQLite tests to the
# Codex command-host console.  A native child failure must become a pytest
# exit code and a log, not a console-control event for the desktop host.
$env:QT_QPA_PLATFORM = "offscreen"
$logRoot = Join-Path $env:TEMP ("everything-finder-pytest-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

# A Qt/native test module can leave process-global state even after its own
# tests pass. If multiple .py targets are supplied, execute each target in a
# fresh interpreter so one module cannot poison the next one.
$targets = @($PytestArgs | Where-Object { $_ -match '\.py(?:::.*)?$' })
$sharedArgs = @($PytestArgs | Where-Object { $_ -notmatch '\.py(?:::.*)?$' })
if ($targets.Count -eq 0) {
    $targets = @('')
}

$overallExitCode = 0
$runIndex = 0
foreach ($target in $targets) {
    $runIndex++
    $stdoutPath = Join-Path $logRoot ("stdout-$runIndex.txt")
    $stderrPath = Join-Path $logRoot ("stderr-$runIndex.txt")
    $arguments = @("-m", "pytest") + $sharedArgs
    if ($target) {
        $arguments += $target
    }
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $workspace `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru `
        -Wait

    Write-Output ("pytest target: " + ($target ? $target : "<arguments>"))
    Write-Output ("pytest exit code: " + $process.ExitCode)
    if ($process.ExitCode -ne 0) {
        $overallExitCode = $process.ExitCode
    }
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath
    }
}
Write-Output ("pytest logs: " + $logRoot)
exit $overallExitCode
