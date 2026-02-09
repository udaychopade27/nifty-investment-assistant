"""Paper trading performance analytics."""
from typing import List, Dict, Any


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [t for t in trades if t.get("type") == "close"]
    realized = [t.get("realized_pnl", 0.0) for t in closed]
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p <= 0]
    return {
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed), 3) if closed else 0.0,
        "realized_pnl": round(sum(realized), 2),
        "avg_pnl": round(sum(realized) / len(closed), 2) if closed else 0.0,
    }
