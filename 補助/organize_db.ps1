# ============================================================
#  jmc Decorated Tomb DB - reorganize & rename to English
#  Needs on Desktop: db_map.csv, db_plan.csv
#  Source folder is never modified. Output = new folder.
# ============================================================

# ---- settings ----------------------------------------------
$Src = Join-Path $env:USERPROFILE "Desktop\jmc_装飾古墳DB公開データ"
$Dst = Join-Path $env:USERPROFILE "Desktop\jmc_DecoratedTombDB_EN"

# TEST = $true   /   REAL = $false
$DryRun = $false
# ------------------------------------------------------------

$ErrorActionPreference = "Stop"
$MapCsv  = Join-Path $env:USERPROFILE "Desktop\db_map.csv"
$PlanCsv = Join-Path $env:USERPROFILE "Desktop\db_plan.csv"

if (-not (Test-Path -LiteralPath $Src))     { throw "Source folder not found: $Src" }
if (-not (Test-Path -LiteralPath $MapCsv))  { throw "db_map.csv not found on Desktop" }
if (-not (Test-Path -LiteralPath $PlanCsv)) { throw "db_plan.csv not found on Desktop" }

function Get-Key([string]$s) {
    $segs = $s.Trim() -split "[_\-\s]+"
    $out = @()
    foreach ($g in $segs) {
        if ($g -match "^\d+$") { $t = $g.TrimStart("0"); if ($t -eq "") { $t = "0" }; $out += $t }
        else { $out += $g.ToLower() }
    }
    return ($out -join "_")
}

$tombInfo = @{}
foreach ($r in (Import-Csv -LiteralPath $MapCsv)) {
    $tombInfo[$r.tomb_ja] = @{ pref = $r.pref_en; tomb = $r.tomb_en }
}
$plan = @{}
foreach ($r in (Import-Csv -LiteralPath $PlanCsv)) { $plan[$r.photo_key] = $r }

$used = New-Object 'System.Collections.Generic.HashSet[string]'
$log  = New-Object System.Collections.ArrayList
$nCopy=0; $nLedger=0; $nExtra=0; $nExcluded=0; $nSkipTomb=0

Get-ChildItem -LiteralPath $Src -Recurse -File | ForEach-Object {

    $ext = $_.Extension.ToLower()
    if ($ext -ne ".jpg" -and $ext -ne ".jpeg") { return }
    if ($ext -eq ".jpeg") { $ext = ".jpg" }

    $rel   = $_.FullName.Substring($Src.Length + 1)
    $parts = $rel -split "\\"
    if ($parts.Count -lt 3) { return }

    $info = $tombInfo[$parts[1]]
    if ($null -eq $info) {
        $nSkipTomb++
        [void]$log.Add([pscustomobject]@{ source=$rel; target=""; status="SKIP_folder_not_in_map" })
        return
    }

    $stem = [IO.Path]::GetFileNameWithoutExtension($_.Name)
    $key  = Get-Key $stem
    $row  = $plan[$key]

    if ($null -ne $row -and $row.decision -eq "exclude") {
        $nExcluded++
        [void]$log.Add([pscustomobject]@{ source=$rel; target=""; status="SKIP_excluded_by_ledger" })
        return
    }

    if ($null -ne $row) {
        $folder = $row.new_folder
        $name   = $row.new_name + $ext
        $status = "OK_ledger"
        $nLedger++
    }
    else {
        $folder = "$($info.pref)\$($info.pref)_$($info.tomb)"
        if ($stem -match "(\d+)\s*$") {
            $num = $Matches[1].TrimStart("0"); if ($num -eq "") { $num = "0" }
            $name = "$($info.pref)_$($info.tomb)_$num$ext"
            $status = "OK_extra"
        } else {
            $name = "$($info.pref)_$($info.tomb)_x$ext"
            $status = "OK_extra_nonum"
        }
        $nExtra++
    }

    $k = "$folder\$name".ToLower()
    $i = 1
    while (-not $used.Add($k)) {
        $base = [IO.Path]::GetFileNameWithoutExtension($name) -replace "_dup\d+$", ""
        $name = "${base}_dup$i$ext"
        $k = "$folder\$name".ToLower()
        $i++
    }

    [void]$log.Add([pscustomobject]@{ source=$rel; target="$folder\$name"; status=$status })

    if (-not $DryRun) {
        $outDir = Join-Path $Dst $folder
        if (-not (Test-Path -LiteralPath $outDir)) {
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $outDir $name)
    }
    $nCopy++
}

$logPath = Join-Path $env:USERPROFILE ("Desktop\run_log_" + (Get-Date -Format "HHmmss") + ".csv")
$log | Export-Csv -LiteralPath $logPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "==================================================="
if ($DryRun) { Write-Host "  TEST RUN - no files were created" -ForegroundColor Yellow }
else         { Write-Host "  DONE" -ForegroundColor Green }
Write-Host "---------------------------------------------------"
Write-Host ("  copied              : {0}" -f $nCopy)
Write-Host ("    from ledger       : {0}   (expect approx 716)" -f $nLedger)
Write-Host ("    donated / no entry: {0}   (expect approx 411)" -f $nExtra)
Write-Host ("  skipped by ledger   : {0}" -f $nExcluded)
Write-Host ("  skipped folder      : {0}" -f $nSkipTomb)
Write-Host ("  output              : {0}" -f $Dst)
Write-Host ("  log                 : {0}" -f $logPath)
Write-Host "==================================================="
Write-Host ""
Write-Host 'If numbers look right, set $DryRun = $false (line 12) and run again.'
