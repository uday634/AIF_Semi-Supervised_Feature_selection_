# Check experiment status and notify when complete
# Run this periodically or set as scheduled task

$processId = 21268
$processName = "python"

# Check if process is running
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue

if ($process) {
    $runtime = (Get-Date) - $process.StartTime
    Write-Host "Experiment is running..." -ForegroundColor Yellow
    Write-Host "  Process ID: $processId" -ForegroundColor Gray
    Write-Host "  Runtime: $([math]::Round($runtime.TotalMinutes, 1)) minutes" -ForegroundColor Gray
    
    # Check if CSV exists (experiment might be progressing)
    if (Test-Path "aif_vs_logistic_vs_anova_10runs.csv") {
        $csvTime = (Get-Item "aif_vs_logistic_vs_anova_10runs.csv").LastWriteTime
        $timeSinceUpdate = (Get-Date) - $csvTime
        Write-Host "  CSV last updated: $([math]::Round($timeSinceUpdate.TotalMinutes, 1)) minutes ago" -ForegroundColor Gray
    }
} else {
    # Process completed - send notification
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "  EXPERIMENT COMPLETED!" -ForegroundColor Green -BackgroundColor Black
    Write-Host ("=" * 70) -ForegroundColor Green
    
    # Check output files
    $outputFiles = @()
    $files = @(
        "aif_vs_logistic_vs_anova_10runs.csv",
        "auc_comparison_10runs_errorbars.png",
        "ap_comparison_10runs_errorbars.png",
        "runtime_comparison_10runs_errorbars.png"
    )
    
    foreach ($file in $files) {
        if (Test-Path $file) {
            $info = Get-Item $file
            $outputFiles += "$file ($([math]::Round($info.Length/1KB, 2)) KB)"
        }
    }
    
    if ($outputFiles.Count -gt 0) {
        Write-Host ""
        Write-Host "Output files created:" -ForegroundColor Cyan
        foreach ($file in $outputFiles) {
            Write-Host "  [OK] $file" -ForegroundColor White
        }
    }
    
    # Check directories
    if (Test-Path "feature_boxplots") {
        $boxplots = (Get-ChildItem "feature_boxplots\*.png" -ErrorAction SilentlyContinue).Count
        if ($boxplots -gt 0) {
            Write-Host "  [OK] Feature boxplots: $boxplots files" -ForegroundColor White
        }
    }
    
    if (Test-Path "roc_curves") {
        $rocs = (Get-ChildItem "roc_curves\*.png" -ErrorAction SilentlyContinue).Count
        if ($rocs -gt 0) {
            Write-Host "  [OK] ROC curves: $rocs files" -ForegroundColor White
        }
    }
    
    # Send Windows notification
    try {
        # Method 1: Try Windows 10/11 Toast Notification
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        
        $xml = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>Experiment Complete!</text>
            <text>AIF vs Logistic vs ANOVA experiment finished. $($outputFiles.Count) output files created.</text>
        </binding>
    </visual>
    <audio src="ms-winsoundevent:Notification.Default" />
</toast>
"@
        
        $toastXml = [Windows.Data.Xml.Dom.XmlDocument]::new()
        $toastXml.LoadXml($xml)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)
        $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Experiment Monitor')
        $notifier.Show($toast)
        
        Write-Host ""
        Write-Host "[OK] Windows notification sent!" -ForegroundColor Green
    } catch {
        # Method 2: Fallback - System beep
        Write-Host ""
        Write-Host "Sending audio notification..." -ForegroundColor Yellow
        [System.Console]::Beep(800, 300)
        Start-Sleep -Milliseconds 200
        [System.Console]::Beep(1000, 300)
        Start-Sleep -Milliseconds 200
        [System.Console]::Beep(1200, 500)
        Write-Host "[OK] Audio notification sent!" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Experiment summary complete. Check the output files!" -ForegroundColor Cyan
}
