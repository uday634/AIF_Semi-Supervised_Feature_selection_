# Simple experiment monitor - checks every minute until completion
# Usage: .\monitor.ps1 [processId]

param(
    [int]$ProcessId = 21268
)

Write-Host "Monitoring experiment (PID: $ProcessId)..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Yellow

while ($true) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    
    if (-not $process) {
        Write-Host "`n" -NoNewline
        Write-Host ("=" * 60) -ForegroundColor Green
        Write-Host "  EXPERIMENT COMPLETED!" -ForegroundColor Green
        Write-Host ("=" * 60) -ForegroundColor Green
        
        # Check files
        $files = @(
            "aif_vs_logistic_vs_anova_10runs.csv",
            "auc_comparison_10runs_errorbars.png",
            "ap_comparison_10runs_errorbars.png",
            "runtime_comparison_10runs_errorbars.png"
        )
        
        Write-Host "`nOutput files:" -ForegroundColor Cyan
        foreach ($file in $files) {
            if (Test-Path $file) {
                Write-Host "  [OK] $file" -ForegroundColor White
            }
        }
        
        # Send notification
        try {
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
            
            $xml = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>Experiment Complete!</text>
            <text>AIF experiment finished. Check output files.</text>
        </binding>
    </visual>
    <audio src="ms-winsoundevent:Notification.Default" />
</toast>
"@
            $toastXml = [Windows.Data.Xml.Dom.XmlDocument]::new()
            $toastXml.LoadXml($xml)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($toastXml)
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('AIF Experiment')
            $notifier.Show($toast)
            Write-Host "`n[OK] Notification sent!" -ForegroundColor Green
        } catch {
            [System.Console]::Beep(800, 300)
            Start-Sleep -Milliseconds 200
            [System.Console]::Beep(1000, 300)
            Start-Sleep -Milliseconds 200
            [System.Console]::Beep(1200, 500)
            Write-Host "`n[OK] Audio notification sent!" -ForegroundColor Green
        }
        
        break
    }
    
    $runtime = (Get-Date) - $process.StartTime
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Running... ($([math]::Round($runtime.TotalMinutes, 1)) min)" -ForegroundColor Gray
    Start-Sleep -Seconds 60
}
