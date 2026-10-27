# Daily security-news run: collect -> build the Korean "AI security newspaper" HTML edition.
# Usage (repo root, any shell):  powershell -NoProfile -File scripts\security_feed\run_daily.ps1
# Output: ..\jarvis_drafts\security_paper\YYYY-MM-DD.html  (outside the repo, public data only)

$ErrorActionPreference = "Stop"

# ===== Paths =====
$Root  = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Py    = Join-Path $Root ".venv\Scripts\python.exe"
$Today = Get-Date -Format "yyyy-MM-dd"
$News  = Join-Path $Root "data\security\news_$Today.json"   # data/ is gitignored
$Out   = Join-Path (Split-Path $Root -Parent) "jarvis_drafts\security_paper"

New-Item -ItemType Directory -Force (Split-Path $News) | Out-Null

# ===== Run =====
# collect.py skips URLs already seen; --merge keeps earlier items of the same day, so reruns add to the edition.
& $Py (Join-Path $PSScriptRoot "collect.py") --out $News --merge
if ($LASTEXITCODE -ne 0) { throw "collect.py failed ($LASTEXITCODE)" }

# --edition security sets the Korean title inside Python (PowerShell 5.1 mangles non-ASCII arguments)
& $Py (Join-Path $Root "scripts\morning_paper.py") --news $News --out $Out --edition security
if ($LASTEXITCODE -ne 0) { throw "morning_paper.py failed ($LASTEXITCODE)" }
