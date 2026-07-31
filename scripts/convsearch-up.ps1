#requires -version 5
<#
.SYNOPSIS
  Start convsearch locally, creating and optionally populating a workspace first.

.EXAMPLE
  powershell -File scripts/convsearch-up.ps1
  powershell -File scripts/convsearch-up.ps1 .\workspace .\chatgpt-export.zip

.DESCRIPTION
  The optional export ZIP can also be set in CONVSEARCH_EXPORT_ZIP (or
  CHATGPT_EXPORT_ZIP). The Chrome extension uses only the local server at
  http://127.0.0.1:8756.
#>
[CmdletBinding()]
param(
    [string]$Workspace = "./workspace",
    [string]$ExportZip = ""
)

$ErrorActionPreference = "Stop"
if (-not $ExportZip) {
    $ExportZip = if ($env:CONVSEARCH_EXPORT_ZIP) { $env:CONVSEARCH_EXPORT_ZIP } else { $env:CHATGPT_EXPORT_ZIP }
}
$DbPath = Join-Path $Workspace "database/convsearch.sqlite3"

# In a checkout with a uv project, use its locked environment. --no-sync avoids
# waiting on the uv lock when a previously started server is still running.
if (((Test-Path -LiteralPath "pyproject.toml" -PathType Leaf) -or (Test-Path -LiteralPath ".venv" -PathType Container)) -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    $Convsearch = @("uv", "run", "--no-sync", "convsearch")
} elseif (Get-Command convsearch -ErrorAction SilentlyContinue) {
    $Convsearch = @("convsearch")
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    $Convsearch = @("uv", "run", "--no-sync", "convsearch")
} else {
    Write-Error "Neither 'uv' nor 'convsearch' is available on PATH. Install the engine first: uv sync --extra ml --extra llm"
    exit 1
}

function Invoke-Convsearch {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $CommandArgs = @()
    if ($Convsearch.Length -gt 1) { $CommandArgs += $Convsearch[1..($Convsearch.Length - 1)] }
    $CommandArgs += $Arguments
    & $Convsearch[0] $CommandArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath $DbPath -PathType Leaf)) {
    if ($ExportZip -and -not (Test-Path -LiteralPath $ExportZip -PathType Leaf)) {
        Write-Error "ChatGPT export ZIP not found: $ExportZip"
        exit 1
    }
    Write-Host "No workspace database found; initializing: $Workspace"
    Invoke-Convsearch init $Workspace
    if ($ExportZip) {
        Write-Host "Importing ChatGPT export: $ExportZip"
        Invoke-Convsearch import $ExportZip -w $Workspace
        Write-Host "Building the local search index..."
        Invoke-Convsearch index -w $Workspace
    } else {
        $cmd = $Convsearch -join " "
        Write-Host "Workspace is empty. To add history later, run:"
        Write-Host "  $cmd import <your-ChatGPT-export.zip> -w `"$Workspace`""
        Write-Host "  $cmd index -w `"$Workspace`""
    }
} else {
    Write-Host "Using existing workspace: $Workspace"
}

Write-Host "Starting local convsearch server: http://127.0.0.1:8756"
Write-Host "Next: load the unpacked extension/ folder in chrome://extensions, then open its side panel."
Write-Host "Press Ctrl+C to stop."
Invoke-Convsearch serve -w $Workspace
