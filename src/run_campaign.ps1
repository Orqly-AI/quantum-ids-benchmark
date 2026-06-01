# Detached, self-healing launcher for the quantum-ids experiment campaign.
# Runs in its OWN terminal window, fully independent of any Claude Code session:
# closing Claude, switching accounts, or ending the session does NOT affect it.
# Sleep is disabled on AC so WSL stays up. The campaign (overnight.sh) is
# idempotent + per-run resumable, so if WSL ever drops this loop relaunches and
# continues exactly where it left off. Keep this window open until it prints DONE.
$ErrorActionPreference = "Continue"
$res = "/mnt/c/Research work 2/quantum-ids/results/.overnight/ALL_COMPLETE"
Write-Host "================ quantum-ids campaign (detached terminal) ================"
Write-Host "Started: $(Get-Date)"
for ($i = 1; $i -le 300; $i++) {
    $done = wsl -d Ubuntu-22.04 -- bash -lc "ls '$res' 2>/dev/null"
    if ($done) { Write-Host "`n=== ALL_COMPLETE detected. Campaign finished. ==="; break }
    Write-Host "`n[attempt $i $(Get-Date -Format HH:mm:ss)] launching/resuming overnight.sh ..."
    wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' < '/mnt/c/Research work 2/quantum-ids/src/overnight.sh' > ~/overnight.sh && bash ~/overnight.sh"
    Write-Host "[attempt $i] overnight.sh returned (exit/hiccup); re-checking completion ..."
    Start-Sleep -Seconds 8
}
Write-Host "`nLauncher loop ended at $(Get-Date). This window can be closed."
Write-Host "Results: C:\Research work 2\quantum-ids\results\  (summary_agg.csv, summary.tex, figures\)"
