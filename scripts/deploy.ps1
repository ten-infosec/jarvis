# ===== Jarvis cloud deploy: test -> build -> push -> deploy -> smoke check =====
# Run from the repo root in PowerShell:
#   .\scripts\deploy.ps1 -Version 0.4            # real deploy
#   .\scripts\deploy.ps1 -Version 0.4 -DryRun    # only print the commands, change nothing
# This file is ASCII only on purpose: Windows PowerShell 5.1 misreads non-ASCII text in a .ps1 saved without BOM.

param(
    # Image version. Only digits like 0.4 or 0.4.1 are accepted (blocks typos and odd characters)
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+(\.\d+)?$')]
    [string]$Version,

    # Print each command without running it
    [switch]$DryRun,

    # Allow deploying with uncommitted changes (normally refused so the image always matches a commit)
    [switch]$AllowDirty
)

# Stop the script on any PowerShell error
$ErrorActionPreference = "Stop"

# ===== Settings (no secrets here: the API key stays in Secret Manager) =====
$Project   = "jarvis-rag-511912"
$Region    = "asia-northeast3"
$Service   = "jarvis-rag"
$Registry  = "jarvis"
$LocalTag  = "jarvis-cloud:$Version"
$RemoteTag = "$Region-docker.pkg.dev/$Project/$Registry/jarvis-cloud:$Version"

# ===== Helper: print a step, run it, stop if it failed =====
# Arguments are passed as an array, one element per argument.
# That way PowerShell never re-splits a value at a comma or space (the 2026-10-17 incident).
function Invoke-Step {
    param([string]$Title, [string]$Exe, [string[]]$Arguments)
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    Write-Host "    $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $Exe @Arguments
    # Native programs (docker, gcloud, python) report failure through the exit code, not an exception
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title (exit code $LASTEXITCODE)" }
}

# ===== 0. Pre-checks =====
# Must run from the repo root, where Dockerfile.cloud lives
if (-not (Test-Path "Dockerfile.cloud")) { throw "Run this from the repo root (C:\dev\jarvis)." }

# The three tools this script calls must be installed (otherwise later checks could give misleading results)
foreach ($tool in @("git", "python", "docker", "gcloud")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found on PATH." }
}

# The cloud build must use the allowlist ignore file, so .env and private data never enter the build context
if (-not (Test-Path "Dockerfile.cloud.dockerignore")) { throw "Dockerfile.cloud.dockerignore missing: refusing to build (could leak .env / private data)." }

# The public paper index must exist, otherwise the image would ship without search data
if (-not (Test-Path "data\arxiv\chroma")) { throw "data\arxiv\chroma not found. Build the arXiv index first." }

# Uncommitted changes mean the deployed image would not match any commit
$dirty = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Not a git repository (git status failed)." }
if ($dirty -and -not $AllowDirty) {
    Write-Host $dirty
    throw "Uncommitted changes. Commit first, or pass -AllowDirty if you really mean it."
}

# Versions are never reused: refuse if this tag is already in Artifact Registry
if (-not $DryRun) {
    # gcloud prints "not found" on stderr when the tag is missing; that is the good case here.
    # ErrorActionPreference is relaxed for this one call because Windows PowerShell 5.1
    # turns redirected stderr of a native program into a terminating error.
    $ErrorActionPreference = "Continue"
    $out = & gcloud artifacts docker images describe $RemoteTag --project $Project --format "value(image_summary.digest)" 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -eq 0) { throw "Tag $Version already exists in Artifact Registry. Use a new version number." }
    # Any failure other than "not found" (expired login, network, wrong project) stops the deploy
    if ($out -notmatch "NOT_FOUND|not found") { throw "Could not check the tag (login or network problem?): $out" }
}

# ===== 1. Tests (no API key, no network) =====
Invoke-Step "Run tests" "python" @("-m", "pytest", "tests", "-q")

# ===== 2. Build -> tag -> push =====
Invoke-Step "Build image"   "docker" @("build", "-f", "Dockerfile.cloud", "-t", $LocalTag, ".")
Invoke-Step "Tag image"     "docker" @("tag", $LocalTag, $RemoteTag)
Invoke-Step "Push image"    "docker" @("push", $RemoteTag)

# ===== 3. Deploy (image only; env vars, secrets, limits and service account are kept) =====
Invoke-Step "Deploy to Cloud Run" "gcloud" @("run", "deploy", $Service, "--image", $RemoteTag, "--region", $Region, "--project", $Project)

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run finished. Nothing was changed." -ForegroundColor Yellow
    exit 0
}

# ===== 4. Smoke check =====
$url = (& gcloud run services describe $Service --region $Region --project $Project --format "value(status.url)").Trim()
Write-Host ""
Write-Host "==> Smoke check $url" -ForegroundColor Cyan

# /health must answer (the first call may take a few seconds because of the cold start)
$health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 60
Write-Host "    /health -> $($health | ConvertTo-Json -Compress)"

# /docs must be hidden in cloud mode (expect 404)
try {
    $r = Invoke-WebRequest -Uri "$url/docs" -UseBasicParsing -TimeoutSec 30
    throw "/docs -> $($r.StatusCode): docs page is public in cloud mode"
} catch {
    # No response at all (network error) leaves Response empty: treat it as a failed check
    if ($null -eq $_.Exception.Response) { throw "/docs check failed: no response ($($_.Exception.Message))" }
    $code = [int]$_.Exception.Response.StatusCode
    if ($code -eq 404) { Write-Host "    /docs -> 404 (hidden, OK)" -ForegroundColor Green }
    else { throw "/docs -> $code (expected 404)" }
}

Write-Host ""
Write-Host "Deployed $RemoteTag" -ForegroundColor Green
Write-Host "Next: ask one question in the web chat, then check the log (xff_hops should be 1):"
Write-Host '  gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.message:ask" --limit 5 --format="value(timestamp,jsonPayload.status,jsonPayload.xff_hops)"'
