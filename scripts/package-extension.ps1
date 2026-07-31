<#
.SYNOPSIS
    Builds a Chrome Web Store-ready ZIP of the convsearch extension.

.DESCRIPTION
    The Chrome Web Store requires manifest.json at the ROOT of the ZIP, so we
    archive the *contents* of extension/ (not the extension/ folder itself).
    OS cruft (.DS_Store, Thumbs.db) and *.map files are excluded.

    Output: dist/convsearch-extension-v<version>.zip

.EXAMPLE
    ./scripts/package-extension.ps1
#>

$ErrorActionPreference = 'Stop'

# Resolve repo root (this script lives in scripts/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

$ExtDir   = Join-Path $RepoRoot 'extension'
$DistDir  = Join-Path $RepoRoot 'dist'
$Manifest = Join-Path $ExtDir  'manifest.json'

if (-not (Test-Path $Manifest)) {
    throw "manifest not found at $Manifest"
}

# Read the version from manifest.json.
$Version = (Get-Content -Raw -Path $Manifest | ConvertFrom-Json).version
if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "could not read version from $Manifest"
}

$ZipName = "convsearch-extension-v$Version.zip"
$ZipPath = Join-Path $DistDir $ZipName

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

Write-Host "Packaging convsearch extension v$Version"
Write-Host "  source: $ExtDir"
Write-Host "  output: $ZipPath"

# Stage the extension contents into a temp dir, dropping excluded files, so the
# archive has manifest.json at its root and no OS cruft / source maps.
$Stage = Join-Path ([System.IO.Path]::GetTempPath()) ("convsearch-pkg-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
try {
    Copy-Item -Path (Join-Path $ExtDir '*') -Destination $Stage -Recurse -Force

    Get-ChildItem -Path $Stage -Recurse -File -Force |
        Where-Object {
            $_.Name -eq '.DS_Store' -or
            $_.Name -eq 'Thumbs.db' -or
            $_.Extension -eq '.map'
        } |
        Remove-Item -Force

    # Compress the CONTENTS of the stage dir (note the \* ) so manifest.json is
    # at the ZIP root, not nested under an extension/ folder.
    Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $ZipPath -Force
}
finally {
    Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
}

# Report the result: path, size, and top-level file count.
$SizeBytes = (Get-Item $ZipPath).Length
$Archive   = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $entryCount   = $Archive.Entries.Count
    $manifestRoot = [bool]($Archive.Entries | Where-Object { $_.FullName -eq 'manifest.json' })
}
finally {
    $Archive.Dispose()
}

Write-Host ""
Write-Host "Done."
Write-Host "  zip:   $ZipPath"
Write-Host "  size:  $SizeBytes bytes"
Write-Host "  files: $entryCount"
Write-Host "  manifest at root: $manifestRoot"
