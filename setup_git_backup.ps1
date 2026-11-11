# SuperRR GitHub backup automation
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_git_backup.ps1

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = 'D:\SuperRR'
$OutputDirectory = Join-Path $Root 'OUTPUT'
$OutputFiles = @(
    'top_10_batters_pressure.csv',
    'top_10_bowlers_pressure.csv',
    'full_player_pressure_metrics.csv'
)

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    & git @Arguments
    $exitCode = $LASTEXITCODE
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "Git command failed with exit code ${exitCode}: git $($Arguments -join ' ')"
    }
    return $exitCode
}

function Require-Path {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "Workspace directory does not exist: $Root"
    }

    Set-Location -LiteralPath $Root

    Write-Host "`n[1/7] Creating root .gitignore..." -ForegroundColor Cyan
    $GitIgnore = @(
        '# Python environments and caches'
        '.venv/'
        '__pycache__/'
        '*.pyc'
        ''
        '# Raw datasets and generated CSVs outside OUTPUT/'
        'data/'
        '*.csv'
        '!OUTPUT/'
        '!OUTPUT/*.csv'
        ''
        '# Local/editor/system files'
        '.DS_Store'
        'Thumbs.db'
        '.vscode/'
    )
    $GitIgnore | Out-File -LiteralPath (Join-Path $Root '.gitignore') -Encoding utf8

    Write-Host "[2/7] Verifying GitHub Pages dashboard location..." -ForegroundColor Cyan
    $RootIndex = Join-Path $Root 'index.html'
    if (-not (Test-Path -LiteralPath $RootIndex -PathType Leaf)) {
        $Candidates = @(Get-ChildItem -LiteralPath $Root -Filter 'index.html' -File -Recurse -ErrorAction SilentlyContinue)
        if ($Candidates.Count -eq 0) {
            throw "index.html was not found in the workspace or any subdirectory."
        }
        if ($Candidates.Count -gt 1) {
            $CandidateList = ($Candidates | ForEach-Object { $_.FullName }) -join '; '
            throw "Multiple index.html files found. Resolve the ambiguity before continuing: $CandidateList"
        }
        Move-Item -LiteralPath $Candidates[0].FullName -Destination $RootIndex -Force
        Write-Host "Moved index.html to $RootIndex" -ForegroundColor Green
    }
    Require-Path -Path $RootIndex -Description 'Root dashboard'

    Write-Host "[3/7] Ensuring OUTPUT contains final CSV exports..." -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    }
    foreach ($FileName in $OutputFiles) {
        $RootFile = Join-Path $Root $FileName
        $OutputFile = Join-Path $OutputDirectory $FileName
        if ((Test-Path -LiteralPath $RootFile -PathType Leaf) -and
            -not (Test-Path -LiteralPath $OutputFile -PathType Leaf)) {
            Move-Item -LiteralPath $RootFile -Destination $OutputFile
        }
        Require-Path -Path $OutputFile -Description "Required output file $FileName"
    }

    Write-Host "[4/7] Initializing Git repository and main branch..." -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath (Join-Path $Root '.git') -PathType Container)) {
        Invoke-Git -Arguments @('init')
    }
    Invoke-Git -Arguments @('branch', '-M', 'main')

    Write-Host "[5/7] Staging only designated project assets..." -ForegroundColor Cyan
    $StagePaths = @(
        '.gitignore'
        'index.html'
        'pressure_analysis.py'
        'verify_pressure_math.py'
        'OUTPUT/top_10_batters_pressure.csv'
        'OUTPUT/top_10_bowlers_pressure.csv'
        'OUTPUT/full_player_pressure_metrics.csv'
    )
    foreach ($Path in $StagePaths) {
        Require-Path -Path (Join-Path $Root $Path) -Description "Required project asset $Path"
    }
    Invoke-Git -Arguments @('add', '--', '.gitignore', 'index.html', 'pressure_analysis.py', 'verify_pressure_math.py', 'OUTPUT')

    Write-Host "[6/7] Checking staged changes..." -ForegroundColor Cyan
    Invoke-Git -Arguments @('status', '--short')
    $StagedDiffExitCode = Invoke-Git -Arguments @('diff', '--cached', '--quiet') -AllowFailure
    if ($StagedDiffExitCode -eq 1) {
        $CommitMessage = 'Initial SuperRR analytics dashboard and pressure model'
        Invoke-Git -Arguments @('commit', '-m', $CommitMessage)
    } elseif ($StagedDiffExitCode -ne 0) {
        throw "Unable to determine whether staged changes exist."
    } else {
        Write-Host 'No staged changes require a new commit.' -ForegroundColor Yellow
    }

    Write-Host "[7/7] Configuring GitHub remote and pushing main..." -ForegroundColor Cyan
    $GitHubUsername = Read-Host 'GitHub Username'
    $RepositoryName = Read-Host 'Repository Name'
    if ([string]::IsNullOrWhiteSpace($GitHubUsername) -or
        [string]::IsNullOrWhiteSpace($RepositoryName)) {
        throw 'GitHub Username and Repository Name are required.'
    }

    $RemoteUrl = "https://github.com/$($GitHubUsername.Trim())/$($RepositoryName.Trim()).git"
    $ExistingOrigin = (& git remote get-url origin 2>$null)
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($ExistingOrigin)) {
        Invoke-Git -Arguments @('remote', 'set-url', 'origin', $RemoteUrl)
    } else {
        Invoke-Git -Arguments @('remote', 'add', 'origin', $RemoteUrl)
    }

    Write-Host "Pushing to $RemoteUrl ..." -ForegroundColor Yellow
    Invoke-Git -Arguments @('push', '-u', 'origin', 'main', '--force')
    Write-Host "`nBackup completed successfully." -ForegroundColor Green
} catch {
    Write-Error ("Backup failed: " + $_.Exception.Message)
    Write-Host 'No destructive cleanup was performed. Check the message above, then rerun the script.' -ForegroundColor Yellow
    exit 1
}
