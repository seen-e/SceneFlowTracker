param(
    [string]$HostName = "root@39.101.70.188",
    [int]$Port = 1010,
    [string]$ProjectDir = "/mnt/workspace/SceneFlowTracker",
    [string]$RemoteScript = "/mnt/workspace/SceneFlowTracker/example/watchdog_restart_remote.sh",
    [int]$IntervalSeconds = 60,
    [int]$MountRecoveryTimeoutSeconds = 900,
    [string]$MountCheckPath = "/mnt/data/chachaxu/dataset",
    [string]$KillPattern = "[p]ython"
)

$ErrorActionPreference = "Continue"

Write-Host "SceneFlowTracker local watchdog started."
Write-Host "Remote: ${HostName}:${Port}"
Write-Host "Project: ${ProjectDir}"
Write-Host "Mount check: ${MountCheckPath}"
Write-Host "Kill pattern on failure: pkill -f ${KillPattern}"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] checking remote task..."

    $remoteCommand = "PROJECT_DIR='$ProjectDir' MOUNT_CHECK_PATH='$MountCheckPath' MOUNT_RECOVERY_TIMEOUT_SECONDS='$MountRecoveryTimeoutSeconds' KILL_PATTERN='$KillPattern' bash '$RemoteScript' --once"

    ssh -p $Port $HostName $remoteCommand

    Start-Sleep -Seconds $IntervalSeconds
}
