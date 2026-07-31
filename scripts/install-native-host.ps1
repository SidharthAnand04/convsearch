#requires -version 5
<#
.SYNOPSIS
  One-time setup so the convsearch Chrome extension can auto-start the local server
  via Chrome Native Messaging.

.DESCRIPTION
  After loading the unpacked extension in chrome://extensions, copy its ID and run:

    powershell -ExecutionPolicy Bypass -File scripts/install-native-host.ps1 -ExtensionId <id> -Workspace .\workspace

  This script:
    1. Resolves how to run convsearch (uv-managed project, or a bare `convsearch`).
    2. Writes scripts/native-host/host-config.json (server command, workspace, port, log).
    3. Generates scripts/native-host/convsearch-host.bat (the launcher Chrome runs).
    4. Renders scripts/native-host/com.convsearch.host.json from the template.
    5. Registers the host for Chrome and Edge under HKCU.

  It is idempotent: re-running it simply overwrites the generated files and refreshes
  the registry values.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/install-native-host.ps1 -ExtensionId abcdefghijklmnopabcdefghijklmnop -Workspace .\workspace
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExtensionId,

    [string]$Workspace = "$PSScriptRoot\..\workspace",

    [int]$Port = 8756
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #

$ScriptsDir  = $PSScriptRoot
$RepoRoot    = (Resolve-Path (Join-Path $ScriptsDir "..")).Path
$HostDir     = Join-Path $ScriptsDir "native-host"
$NativeHostPy = Join-Path $ScriptsDir "native_host.py"

if (-not (Test-Path -LiteralPath $HostDir -PathType Container)) {
    New-Item -ItemType Directory -Path $HostDir -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $NativeHostPy -PathType Leaf)) {
    Write-Error "native_host.py not found at $NativeHostPy"
    exit 1
}

# Normalise the extension id (Chrome shows lowercase a-p characters).
$ExtensionId = $ExtensionId.Trim()
if ($ExtensionId -notmatch '^[a-p]{32}$') {
    Write-Warning "Extension id '$ExtensionId' does not look like a 32-char Chrome id; continuing anyway."
}

# Resolve workspace to an absolute path (it need not exist yet).
try {
    $WorkspaceAbs = (Resolve-Path -LiteralPath $Workspace -ErrorAction Stop).Path
} catch {
    $WorkspaceAbs = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Workspace))
}

# Log dir: repo tmp/. Create it so the host can always append.
$LogDir  = Join-Path $RepoRoot "tmp"
if (-not (Test-Path -LiteralPath $LogDir -PathType Container)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogPath = Join-Path $LogDir "native-host.log"

# --------------------------------------------------------------------------- #
# resolve how to run convsearch (the server) and python (the host)            #
# --------------------------------------------------------------------------- #

$HasUv       = [bool](Get-Command uv -ErrorAction SilentlyContinue)
$HasProject  = (Test-Path -LiteralPath (Join-Path $RepoRoot "pyproject.toml")) -or `
               (Test-Path -LiteralPath (Join-Path $RepoRoot ".venv"))
$HasConvsearch = [bool](Get-Command convsearch -ErrorAction SilentlyContinue)

if ($HasUv -and $HasProject) {
    # uv-managed checkout: use the locked env. --no-sync avoids blocking on the lock
    # when a server is already running.
    $ServerCommand = @("uv", "run", "--no-sync", "convsearch")
    $Interpreter   = 'uv run --no-sync python'
} elseif ($HasConvsearch) {
    $ServerCommand = @("convsearch")
    # Find a python for the host launcher.
    $Py = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $Py) { $Py = (Get-Command python3 -ErrorAction SilentlyContinue) }
    if (-not $Py) {
        Write-Error "convsearch is on PATH but no 'python' interpreter was found to run the native host."
        exit 1
    }
    $Interpreter = '"' + $Py.Source + '"'
} elseif ($HasUv) {
    $ServerCommand = @("uv", "run", "--no-sync", "convsearch")
    $Interpreter   = 'uv run --no-sync python'
} else {
    Write-Error "Neither 'uv' nor 'convsearch' is available on PATH. Install the engine first: uv sync --extra ml --extra llm"
    exit 1
}

# --------------------------------------------------------------------------- #
# 1) host-config.json                                                          #
# --------------------------------------------------------------------------- #

$ConfigPath = Join-Path $HostDir "host-config.json"
$Config = [ordered]@{
    command   = $ServerCommand
    workspace = $WorkspaceAbs
    port      = $Port
    log       = $LogPath
}
# ConvertTo-Json wraps a single-element array oddly; -Depth keeps the array intact.
$Config | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
Write-Host "Wrote host config:        $ConfigPath"

# --------------------------------------------------------------------------- #
# 2) convsearch-host.bat launcher                                              #
# --------------------------------------------------------------------------- #

$LauncherPath = Join-Path $HostDir "convsearch-host.bat"
# cd to the repo root first so `uv run` (and relative resolution) works regardless of
# the working directory Chrome hands the launcher.
$LauncherLines = @(
    '@echo off',
    'REM Generated by install-native-host.ps1 - do not edit by hand.',
    ('cd /d "' + $RepoRoot + '"'),
    ($Interpreter + ' "' + $NativeHostPy + '" %*')
)
# Native messaging launchers must be ASCII/ANSI without a BOM.
[System.IO.File]::WriteAllLines($LauncherPath, $LauncherLines, (New-Object System.Text.ASCIIEncoding))
Write-Host "Wrote launcher:           $LauncherPath"

# --------------------------------------------------------------------------- #
# 3) render com.convsearch.host.json                                           #
# --------------------------------------------------------------------------- #

$TemplatePath = Join-Path $HostDir "com.convsearch.host.json.template"
$ManifestPath = Join-Path $HostDir "com.convsearch.host.json"
if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
    Write-Error "Manifest template not found at $TemplatePath"
    exit 1
}
$Template = Get-Content -LiteralPath $TemplatePath -Raw
# JSON needs backslashes escaped in the path string.
$HostPathJson = $LauncherPath -replace '\\', '\\'
$Manifest = $Template.Replace('__HOST_PATH__', $HostPathJson).Replace('__EXTENSION_ID__', $ExtensionId)
# Write without BOM so Chrome's JSON parser is happy.
[System.IO.File]::WriteAllText($ManifestPath, $Manifest, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote host manifest:      $ManifestPath"

# --------------------------------------------------------------------------- #
# 4) register under HKCU for Chrome and Edge                                   #
# --------------------------------------------------------------------------- #

function Register-Host {
    param([string]$BaseKey, [string]$BrowserName)
    $Key = Join-Path $BaseKey "com.convsearch.host"
    try {
        if (-not (Test-Path -LiteralPath $Key)) {
            New-Item -Path $Key -Force | Out-Null
        }
        # The (Default) value must be the absolute path to the rendered manifest json.
        Set-ItemProperty -LiteralPath $Key -Name "(Default)" -Value $ManifestPath
        Write-Host "Registered for $BrowserName."
    } catch {
        Write-Warning ("Could not register for {0}: {1}" -f $BrowserName, $_.Exception.Message)
    }
}

Register-Host -BaseKey "HKCU:\Software\Google\Chrome\NativeMessagingHosts" -BrowserName "Chrome"
Register-Host -BaseKey "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts" -BrowserName "Edge"

# --------------------------------------------------------------------------- #
# done                                                                         #
# --------------------------------------------------------------------------- #

Write-Host ""
Write-Host "Native messaging host installed." -ForegroundColor Green
Write-Host "  Extension id: $ExtensionId"
Write-Host "  Workspace:    $WorkspaceAbs"
Write-Host "  Port:         $Port"
Write-Host "  Server run:   $($ServerCommand -join ' ')"
Write-Host "  Log:          $LogPath"
Write-Host ""
Write-Host "Next: reload the extension in chrome://extensions (or restart the browser),"
Write-Host "then the extension will auto-start the server whenever it is offline."
Write-Host ""
Write-Host "Note: convsearch must be installed (uv sync) and, for a fresh workspace, the"
Write-Host "history must be imported and indexed first, or the server will serve empty results."
