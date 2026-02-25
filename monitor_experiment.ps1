# Monitor experiment and notify on completion
# Usage: .\monitor_experiment.ps1 <PID>

param(
    [Parameter(Mandatory=$false)]
    [int]$ProcessId = 21268
)

Write-Host "Monitoring experiment process (PID: $ProcessId)..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow

$processName = "python"
$checkInterval = 30  # Check every 30 seconds
$lastOutputTime = Get-Date

while ($true) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    
    if (-not $process) {
        Write-Host "`nProcess completed!" -ForegroundColor Green
        
        # Check if output files exist
        $csvFile = "aif_vs_logistic_vs_anova_10runs.csv"
        $pngFiles = @(
            "auc_comparison_10runs_errorbars.png",
            "ap_comparison_10runs_errorbars.png",
            "runtime_comparison_10runs_errorbars.png"
        )
        
        $filesCreated = @()
        if (Test-Path $csvFile) {
            $filesCreated += $csvFile
        }
        foreach ($png in $pngFiles) {
            if (Test-Path $png) {
                $filesCreated += $png
            }
        }
        
        # Send Windows notification
        $title = "Experiment Complete!"
        $body = "AIF vs Logistic vs ANOVA experiment has finished."
        if ($filesCreated.Count -gt 0) {
            $body += "`nCreated files: $($filesCreated.Count)"
        }
        
        # Use Windows 10/11 toast notification
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        
        $xml = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>$title</text>
            <text>$body</text>
        </binding>
    </visual>
    <audio src="ms-winsoundevent:Notification.Default" />
</toast>
"@
        
        try {
            $toastXml = [Windows.Data.Xml.Dom.XmlDocument]::new()
            $toastXml.LoadXml($xml)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PowerShell").Show($toast)
            Write-Host "Notification sent!" -ForegroundColor Green
        } catch {
            # Fallback to simpler notification method
            Write-Host "`n$title" -ForegroundColor Cyan
            Write-Host $body -ForegroundColor White
            [System.Console]::Beep(1000, 500)
        }
        
        Write-Host "`nExperiment Summary:" -ForegroundColor Cyan
        if (Test-Path $csvFile) {
            Write-Host "  ✓ CSV results: $csvFile" -ForegroundColor Green
            $csvInfo = Get-Item $csvFile
            Write-Host "    Size: $([math]::Round($csvInfo.Length/1KB, 2)) KB" -ForegroundColor Gray
            Write-Host "    Modified: $($csvInfo.LastWriteTime)" -ForegroundColor Gray
        }
        foreach ($png in $pngFiles) {
            if (Test-Path $png) {
                Write-Host "  ✓ Chart: $png" -ForegroundColor Green
            }
        }
        
        if (Test-Path "feature_boxplots") {
            $boxplotCount = (Get-ChildItem "feature_boxplots\*.png" -ErrorAction SilentlyContinue).Count
            if ($boxplotCount -gt 0) {
                Write-Host "  ✓ Feature boxplots: $boxplotCount files" -ForegroundColor Green
            }
        }
        
        if (Test-Path "roc_curves") {
            $rocCount = (Get-ChildItem "roc_curves\*.png" -ErrorAction SilentlyContinue).Count
            if ($rocCount -gt 0) {
                Write-Host "  ✓ ROC curves: $rocCount files" -ForegroundColor Green
            }
        }
        
        break
    }
    
    # Check if output file was recently modified (experiment is progressing)
    if (Test-Path "aif_vs_logistic_vs_anova_10runs.csv") {
        $fileTime = (Get-Item "aif_vs_logistic_vs_anova_10runs.csv").LastWriteTime
        if ($fileTime -gt $lastOutputTime) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Experiment still running... (CSV updated)" -ForegroundColor Gray
            $lastOutputTime = $fileTime
        }
    }
    
    Start-Sleep -Seconds $checkInterval
}
