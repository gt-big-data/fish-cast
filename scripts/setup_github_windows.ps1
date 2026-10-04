# FishCast: one-time GitHub setup on Windows.
# Started by SETUP_GITHUB.bat in the repository root. Safe to run again if it stops partway.
#  1. Installs Git and GitHub CLI with winget if missing
#  2. Signs you in to GitHub in your browser (you approve it there)
#  3. Commits this folder and pushes it to gt-big-data/fish-cast (asks before replacing main)
#  4. Creates labels, milestones, issues and the project board (scripts/setup_github.py)

$ErrorActionPreference = "Continue"  # native tools (git, gh) write progress to stderr
$Repo = "gt-big-data/fish-cast"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Step "1/4 Checking tools"
if (-not (Have winget)) { throw "winget is missing. Install 'App Installer' from the Microsoft Store, then run this again." }
if (-not (Have git)) { winget install --id Git.Git -e --source winget; Refresh-Path }
if (-not (Have gh))  { winget install --id GitHub.cli -e --source winget; Refresh-Path }
$py = if (Have py) { "py" } elseif (Have python) { "python" } else { $null }
if (-not $py) { winget install --id Python.Python.3.12 -e --source winget; Refresh-Path; $py = "py" }
if (-not (Have git) -or -not (Have gh)) { throw "Git or GitHub CLI installed but not found yet. Close this window and double-click SETUP_GITHUB.bat again." }
Write-Host "git: $(git --version)"; Write-Host "gh: $((gh --version)[0])"

Step "2/4 Signing in to GitHub"
gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "A browser window will open. Copy the one-time code shown here, paste it there, and approve."
  gh auth login --hostname github.com --git-protocol https --web --scopes "project,read:org"
} else {
  gh auth refresh --hostname github.com --scopes "project,read:org"
}
gh auth setup-git
$login = gh api user --jq .login
Write-Host "Signed in as $login"
if (-not (git config --global user.name))  { git config --global user.name  (Read-Host "Your name for git commits") }
if (-not (git config --global user.email)) { git config --global user.email (Read-Host "Your email for git commits") }

Step "3/4 Pushing the code"
if (-not (Test-Path .git)) { git init -b main | Out-Null }
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m "Set up FishCast repository for Fall 2026" | Out-Null }
if (-not (git remote 2>$null | Select-String -Quiet "^origin$")) { git remote add origin "https://github.com/$Repo.git" }
Write-Host "This replaces main on GitHub (it currently has only the two README text files from November)."
$ok = Read-Host "Type YES to push"
if ($ok -ne "YES") { throw "Stopped before pushing. Nothing on GitHub was changed." }
git push -u origin main --force
if ($LASTEXITCODE -ne 0) { throw "Push failed. Check that $login has write access to $Repo." }

Step "4/4 Creating the task board"
& $py -c "import sys" ; if ($LASTEXITCODE -ne 0) { throw "Python not working." }
$start = Read-Host "Monday of week 1 as YYYY-MM-DD (press Enter to skip due dates)"
$setupArgs = @("scripts/setup_github.py", "--repo", $Repo)
if ($start) { $setupArgs += @("--start-date", $start) }
$protect = Read-Host "Require passing tests and one review before merging to main? (y/N, needs admin)"
if ($protect -eq "y") { $setupArgs += "--protect-main" }
& $py @setupArgs
if ($LASTEXITCODE -ne 0) { throw "Board setup stopped. Fix the message above and run SETUP_GITHUB.bat again; finished parts are skipped." }

Write-Host "`nAll done. Repository: https://github.com/$Repo" -ForegroundColor Green
Write-Host "Next: in the project, add a Board view grouped by Status, then assign the 'good first issue' tasks."
