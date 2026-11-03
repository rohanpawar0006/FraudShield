# FraudShield PowerShell Bootstrap Wrapper
param (
    [switch]$Fast,
    [switch]$CIMock,
    [switch]$Seed,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Set-Location $projectRoot

$pyArgs = @("scripts/bootstrap.py")
if ($Fast) { $pyArgs += "--fast" }
if ($CIMock) { $pyArgs += "--ci-mock" }
if ($Seed) { $pyArgs += "--seed" }
if ($Force) { $pyArgs += "--force" }

if (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" $pyArgs
} else {
    & python $pyArgs
}

