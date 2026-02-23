# Benchmark Progress Monitor
# Run this script to check the status of your Phi-3 benchmark

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Phi-3 Benchmark Progress Monitor" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Check if Python benchmark process is running
Write-Host "[1] Process Status:" -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "  ✓ Python process(es) running:" -ForegroundColor Green
    $pythonProcesses | Select-Object Id, @{Name="Runtime";Expression={(Get-Date) - $_.StartTime}}, @{Name="CPU(s)";Expression={[math]::Round($_.CPU, 2)}}, @{Name="Memory(MB)";Expression={[math]::Round($_.WorkingSet64/1MB, 2)}} | Format-Table
} else {
    Write-Host "  ✗ No Python processes running" -ForegroundColor Red
}

# Check latest log file
Write-Host "`n[2] Latest Log Activity:" -ForegroundColor Yellow
$latestLog = Get-ChildItem "results\*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestLog) {
    Write-Host "  Log File: $($latestLog.Name)" -ForegroundColor Cyan
    Write-Host "  Last Updated: $($latestLog.LastWriteTime)" -ForegroundColor Cyan
    Write-Host "  Time Ago: $((Get-Date) - $latestLog.LastWriteTime)" -ForegroundColor Cyan
    Write-Host "`n  Last 10 lines:" -ForegroundColor Cyan
    Get-Content $latestLog.FullName -Tail 10 | ForEach-Object { Write-Host "    $_" }
}

# Check for new result files
Write-Host "`n[3] Completed Experiments:" -ForegroundColor Yellow
$resultFiles = Get-ChildItem "results\*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 10
if ($resultFiles) {
    Write-Host "  Total result files: $($resultFiles.Count)" -ForegroundColor Green
    Write-Host "  Recent results:" -ForegroundColor Cyan
    $resultFiles | Select-Object -First 5 | ForEach-Object {
        $content = Get-Content $_.FullName -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        $status = if ($content.error) { "❌ FAILED" } else { "✓ SUCCESS" }
        Write-Host "    $status - $($_.Name) ($(($_.LastWriteTime).ToString('HH:mm:ss')))" -ForegroundColor $(if ($content.error) { "Red" } else { "Green" })
    }
} else {
    Write-Host "  No result files yet" -ForegroundColor Yellow
}

# Check HuggingFace model cache
Write-Host "`n[4] Model Download Status:" -ForegroundColor Yellow
$hfHome = if ($env:HF_HOME) { $env:HF_HOME } else { "$env:USERPROFILE\.cache\huggingface" }
$phi3Cache = Get-ChildItem "$hfHome\hub" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*hi-3*" -or $_.Name -like "*icrosoft*" }
if ($phi3Cache) {
    Write-Host "  ✓ Phi-3 model found in cache:" -ForegroundColor Green
    $phi3Cache | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
        Write-Host "    $($_.Name)" -ForegroundColor Cyan
        Write-Host "    Size: $([math]::Round($size, 2)) GB" -ForegroundColor Cyan
        Write-Host "    Last Modified: $($_.LastWriteTime)" -ForegroundColor Cyan
    }
} else {
    Write-Host "  ⏳ Model not yet cached (downloading...)" -ForegroundColor Yellow
    Write-Host "  Cache location: $hfHome\hub" -ForegroundColor Gray
}

# Check GPU if available
Write-Host "`n[5] Hardware Status:" -ForegroundColor Yellow
try {
    $gpu = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        Write-Host "  GPU Info:" -ForegroundColor Cyan
        $gpu -split "`n" | ForEach-Object {
            $parts = $_ -split ","
            Write-Host "    GPU: $($parts[0].Trim())" -ForegroundColor Cyan
            Write-Host "    Utilization: $($parts[1].Trim())%" -ForegroundColor Cyan
            Write-Host "    Memory: $($parts[2].Trim()) / $($parts[3].Trim()) MB" -ForegroundColor Cyan
        }
    }
} catch {
    Write-Host "  No NVIDIA GPU detected or nvidia-smi not available" -ForegroundColor Gray
}

# Recommendations
Write-Host "`n[6] Status Summary:" -ForegroundColor Yellow
if ($pythonProcesses) {
    Write-Host "  ✓ Benchmark is RUNNING" -ForegroundColor Green
    Write-Host "  • Monitor this script every few minutes to track progress" -ForegroundColor Cyan
    Write-Host "  • Results will appear in the results/ directory" -ForegroundColor Cyan
    if (-not $phi3Cache) {
        Write-Host "  ⏳ Currently downloading Phi-3 model (~7GB)" -ForegroundColor Yellow
        Write-Host "  • This is a one-time download (5-15 minutes)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ Benchmark is NOT RUNNING" -ForegroundColor Red
    Write-Host "  • Check the log file above for errors" -ForegroundColor Yellow
    Write-Host "  • Restart with: python experiments/run_benchmark.py" -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Refresh: Re-run this script to update status" -ForegroundColor Gray
Write-Host "============================================================`n" -ForegroundColor Cyan
