# PowerShell script to check AI Trading System status

Write-Host "🔍 Checking AI Trading System Status..." -ForegroundColor Cyan
Write-Host "=" * 50

# Check sessions
$response = Invoke-RestMethod -Uri "http://localhost:5000/api/sessions" -Method Get
Write-Host "`n📊 Active Sessions:" -ForegroundColor Green
foreach ($session in $response) {
    Write-Host "  Session $($session.id): $($session.name)" -ForegroundColor Yellow
    Write-Host "    Status: $($session.status)" -ForegroundColor White
    Write-Host "    Algorithm: $($session.algorithm_type)" -ForegroundColor White
    Write-Host "    Progress: $($session.current_episode)/$($session.total_episodes) episodes" -ForegroundColor White
    Write-Host "    Sharpe Ratio: $($session.sharpe_ratio)" -ForegroundColor White
    Write-Host "    Max Drawdown: $($session.max_drawdown)" -ForegroundColor White
}

# Get recent trades for the active session
if ($response.Count -gt 0) {
    $sessionId = $response[0].id
    Write-Host "`n💰 Recent Trades for Session $sessionId`:" -ForegroundColor Green
    
    try {
        $trades = Invoke-RestMethod -Uri "http://localhost:5000/api/sessions/$sessionId/trades" -Method Get
        if ($trades.Count -eq 0) {
            Write-Host "  No trades yet..." -ForegroundColor Gray
        } else {
            foreach ($trade in $trades | Select-Object -First 5) {
                $color = if ($trade.profit_loss -gt 0) { "Green" } else { "Red" }
                Write-Host "  Trade $($trade.id): P&L = $$($trade.profit_loss)" -ForegroundColor $color
            }
        }
    } catch {
        Write-Host "  Could not fetch trades" -ForegroundColor Red
    }
}

Write-Host "`n✅ Dashboard URL: http://localhost:5000/training_dashboard" -ForegroundColor Cyan
Write-Host "📊 Your training is running! Check the dashboard to see live updates." -ForegroundColor Green