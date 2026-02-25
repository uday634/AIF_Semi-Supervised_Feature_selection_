# Setup automatic notification monitoring for experiment
# This creates a scheduled task that checks every 5 minutes

$scriptPath = Join-Path $PSScriptRoot "check_and_notify.ps1"
$taskName = "AIF_Experiment_Monitor"

Write-Host "Setting up experiment notification monitor..." -ForegroundColor Green

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Create action
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$scriptPath`""

# Create trigger (every 5 minutes)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Monitors AIF experiment and sends notification when complete" | Out-Null

Write-Host "✓ Scheduled task created!" -ForegroundColor Green
Write-Host "  Task name: $taskName" -ForegroundColor Gray
Write-Host "  Checks every 5 minutes" -ForegroundColor Gray
Write-Host "`nTo remove the task later, run:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false" -ForegroundColor Gray

# Run once immediately
Write-Host "`nRunning check now..." -ForegroundColor Cyan
& $scriptPath
