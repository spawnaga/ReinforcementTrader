# PowerShell script to get full session details

Write-Host "`n🔍 Getting Full Session Details..." -ForegroundColor Cyan
Write-Host "=" * 70

# Get active sessions
$sessions = Invoke-RestMethod -Uri "http://localhost:5000/api/sessions" -Method Get

if ($sessions.Count -eq 0) {
    Write-Host "`n❌ No active sessions found!" -ForegroundColor Red
    exit
}

$session = $sessions[0]
$sessionId = $session.id

Write-Host "`n📊 Session $sessionId Details:" -ForegroundColor Green
Write-Host "  Name: $($session.name)" -ForegroundColor Yellow
Write-Host "  Status: $($session.status)" -ForegroundColor Yellow
Write-Host "  Algorithm: $($session.algorithm_type)" -ForegroundColor Yellow
Write-Host "  Progress: $($session.current_episode)/$($session.total_episodes) episodes ($([math]::Round($session.current_episode / $session.total_episodes * 100, 1))%)" -ForegroundColor Yellow
Write-Host "  Start Time: $($session.start_time)" -ForegroundColor Yellow

# Get session-specific trades
Write-Host "`n💰 Recent Trades:" -ForegroundColor Green
try {
    $tradesUrl = "http://localhost:5000/api/sessions/$sessionId/trades"
    $trades = Invoke-RestMethod -Uri $tradesUrl -Method Get
    
    if ($trades.Count -eq 0) {
        Write-Host "  No trades executed yet..." -ForegroundColor Gray
    } else {
        $totalPL = 0
        $wins = 0
        $losses = 0
        
        foreach ($trade in $trades | Select-Object -First 10) {
            $pl = [decimal]$trade.profit_loss
            $totalPL += $pl
            
            if ($pl -gt 0) {
                $wins++
                Write-Host "  Trade $($trade.id): +$$pl" -ForegroundColor Green
            } else {
                $losses++
                Write-Host "  Trade $($trade.id): $$pl" -ForegroundColor Red
            }
        }
        
        Write-Host "`n📈 Summary:" -ForegroundColor Cyan
        Write-Host "  Total P&L: $$totalPL" -ForegroundColor $(if ($totalPL -gt 0) { "Green" } else { "Red" })
        Write-Host "  Win Rate: $([math]::Round($wins / ($wins + $losses) * 100, 1))% ($wins wins, $losses losses)" -ForegroundColor White
    }
} catch {
    Write-Host "  Could not fetch trades: $_" -ForegroundColor Red
}

# Get training metrics
Write-Host "`n📊 Training Metrics:" -ForegroundColor Green
try {
    $metricsUrl = "http://localhost:5000/api/sessions/$sessionId/metrics"
    $metrics = Invoke-RestMethod -Uri $metricsUrl -Method Get
    
    if ($metrics.Count -gt 0) {
        $latestMetric = $metrics | Sort-Object -Property timestamp -Descending | Select-Object -First 1
        Write-Host "  Latest Reward: $($latestMetric.reward)" -ForegroundColor Yellow
        Write-Host "  Latest Loss: $($latestMetric.loss)" -ForegroundColor Yellow
    } else {
        Write-Host "  No metrics recorded yet..." -ForegroundColor Gray
    }
} catch {
    # Metrics endpoint might not exist
}

Write-Host "`n✅ Dashboard URL: http://localhost:5000/training_dashboard" -ForegroundColor Cyan
Write-Host "📊 Training Monitor: python training_monitor.py --url http://localhost:5000" -ForegroundColor Cyan
Write-Host "`nNote: If the dashboard shows blank, refresh the page or check the browser console for errors." -ForegroundColor Yellow