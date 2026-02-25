# Simple notification script for experiment completion
# Run this in a separate terminal: .\notify_on_complete.ps1

$processId = 21268
$processName = "python"
$checkInterval = 60  # Check every minute

Write-Host "Monitoring experiment (PID: $processId)..." -ForegroundColor Green
Write-Host "Will notify when experiment completes." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Gray

while ($true) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    
    if (-not $process) {
        # Process completed!
        Write-Host "`n" -NoNewline
        Write-Host "=" * 60 -ForegroundColor Cyan
        Write-Host "EXPERIMENT COMPLETED!" -ForegroundColor Green
        Write-Host "=" * 60 -ForegroundColor Cyan
        
        # Check output files
        $results = @()
        if (Test-Path "aif_vs_logistic_vs_anova_10runs.csv") {
            $results += "✓ Results CSV created"
        }
        if (Test-Path "auc_comparison_10runs_errorbars.png") {
            $results += "✓ AUC comparison chart"
        }
        if (Test-Path "ap_comparison_10runs_errorbars.png") {
            $results += "✓ AP comparison chart"
        }
        if (Test-Path "runtime_comparison_10runs_errorbars.png") {
            $results += "✓ Runtime comparison chart"
        }
        
        Write-Host "`nOutput files:" -ForegroundColor Yellow
        foreach ($r in $results) {
            Write-Host "  $r" -ForegroundColor White
        }
        
        # Windows notification using BurntToast module (if available) or fallback
        try {
            # Try using Windows 10/11 toast notification API
            Add-Type -AssemblyName System.Windows.Forms
            $notification = New-Object System.Windows.Forms.NotifyIcon
            $notification.Icon = [System.Drawing.SystemIcons]::Information
            $notification.BalloonTipTitle = "Experiment Complete!"
            $notification.BalloonTipText = "AIF vs Logistic vs ANOVA experiment has finished.`n$($results.Count) output files created."
            $notification.Visible = $true
            $notification.ShowBalloonTip(10000)
            Start-Sleep -Seconds 2
            $notification.Dispose()
        } catch {
            # Fallback: console beep and message
            [System.Console]::Beep(800, 300)
            [System.Console]::Beep(1000, 300)
            [System.Console]::Beep(1200, 500)
        }
        
        Write-Host "`nNotification sent! Check the output files." -ForegroundColor Green
        break
    }
    
    # Show progress every 5 minutes
    $elapsed = (Get-Date) - $process.StartTime
    if (($elapsed.TotalMinutes % 5) -lt 1) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Still running... ($([math]::Round($elapsed.TotalMinutes, 1)) minutes elapsed)" -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds $checkInterval
}
