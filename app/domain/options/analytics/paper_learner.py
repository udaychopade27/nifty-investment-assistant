"""Online learner for paper-trade outcomes.

This is intentionally lightweight: no external ML dependency and safe default behavior.
It scores signals and updates weights after paper trade closes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def build_features(signal: Dict[str, Any], indicator: Dict[str, Any]) -> Dict[str, float]:
    signal_type = signal.get("signal")
    side_ce = 1.0 if signal_type == "BUY_CE" else 0.0
    side_pe = 1.0 if signal_type == "BUY_PE" else 0.0
    oi_change = indicator.get("oi_change")
    vol_spike = indicator.get("volume_spike")
    atr = indicator.get("atr")
    pcr = indicator.get("pcr")
    rsi = indicator.get("rsi")
    macd_hist = indicator.get("macd_hist")
    boll_pos = indicator.get("boll_pos")
    rr = float(signal.get("rr", 0.0) or 0.0)

    # Basic normalization to keep updates stable.
    atr_n = 0.0 if atr is None else max(-2.0, min(2.0, float(atr) / 20.0))
    oi_n = 0.0 if oi_change is None else max(-2.0, min(2.0, float(oi_change) / 1000000.0))
    pcr_n = 0.0 if pcr is None else max(-2.0, min(2.0, float(pcr) - 1.0))
    rsi_n = 0.0 if rsi is None else max(-2.0, min(2.0, (float(rsi) - 50.0) / 20.0))
    macd_n = 0.0 if macd_hist is None else max(-2.0, min(2.0, float(macd_hist) / 10.0))
    boll_n = 0.0 if boll_pos is None else max(-2.0, min(2.0, (float(boll_pos) - 0.5) * 2.0))
    rr_n = max(0.0, min(3.0, rr / 2.0))

    return {
        "bias": 1.0,
        "side_ce": side_ce,
        "side_pe": side_pe,
        "atr_n": atr_n,
        "oi_n": oi_n,
        "vol_spike": 1.0 if vol_spike is True else 0.0,
        "pcr_n": pcr_n,
        "rsi_n": rsi_n,
        "macd_n": macd_n,
        "boll_n": boll_n,
        "rr_n": rr_n,
    }


class PaperSignalLearner:
    def __init__(
        self,
        enabled: bool,
        model_path: str,
        learning_rate: float = 0.04,
        min_score: float = 0.45,
        warmup_trades: int = 20,
    ):
        self.enabled = bool(enabled)
        self.model_path = Path(model_path)
        self.learning_rate = float(learning_rate)
        self.min_score = float(min_score)
        self.warmup_trades = int(warmup_trades)
        self.samples = 0
        self.wins = 0
        self.weights: Dict[str, float] = {}
        self._load()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "samples": self.samples,
            "wins": self.wins,
            "win_rate": round(self.wins / self.samples, 3) if self.samples else 0.0,
            "min_score": self.min_score,
            "warmup_trades": self.warmup_trades,
            "ready": self.samples >= self.warmup_trades,
        }

    def predict(self, features: Dict[str, float]) -> float:
        if not self.enabled:
            return 0.5
        score = 0.0
        for k, v in features.items():
            score += self.weights.get(k, 0.0) * float(v)
        return self._sigmoid(score)

    def allow_signal(self, features: Dict[str, float]) -> bool:
        if not self.enabled:
            return True
        if self.samples < self.warmup_trades:
            return True
        return self.predict(features) >= self.min_score

    def observe(self, features: Dict[str, float], won: bool) -> None:
        if not self.enabled:
            return
        y = 1.0 if won else 0.0
        p = self.predict(features)
        err = y - p
        lr = self.learning_rate
        for k, x in features.items():
            w = self.weights.get(k, 0.0)
            self.weights[k] = w + (lr * err * float(x))
        self.samples += 1
        if won:
            self.wins += 1
        self._save()

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _load(self) -> None:
        if not self.model_path.exists():
            self.weights = {}
            return
        try:
            data = json.loads(self.model_path.read_text())
            self.weights = {k: float(v) for k, v in (data.get("weights") or {}).items()}
            self.samples = int(data.get("samples", 0))
            self.wins = int(data.get("wins", 0))
        except Exception:
            self.weights = {}
            self.samples = 0
            self.wins = 0

    def _save(self) -> None:
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "weights": self.weights,
                "samples": self.samples,
                "wins": self.wins,
            }
            self.model_path.write_text(json.dumps(payload))
        except Exception:
            # Non-fatal in production runtime; learning continues in memory.
            return

    def train_from_samples(self, samples_path: str) -> Dict[str, Any]:
        path = Path(samples_path)
        if not self.enabled:
            return {"trained": False, "reason": "ml_disabled"}
        if not path.exists():
            return {"trained": False, "reason": "samples_not_found"}
        count = 0
        wins = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                feats = row.get("features") or {}
                won = bool(row.get("won", False))
            except Exception:
                continue
            self.observe(feats, won=won)
            count += 1
            if won:
                wins += 1
        return {
            "trained": True,
            "samples_used": count,
            "wins": wins,
            "win_rate": round((wins / count) * 100.0, 2) if count else 0.0,
            "status": self.status(),
        }

    def walk_forward_evaluate(self, samples_path: str, train_ratio: float = 0.7) -> Dict[str, Any]:
        path = Path(samples_path)
        if not path.exists():
            return {"ok": False, "reason": "samples_not_found"}
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ts = row.get("ts")
                dt = datetime.fromisoformat(str(ts)) if ts else datetime.min
                rows.append((dt, row))
            except Exception:
                continue
        if len(rows) < 20:
            return {"ok": False, "reason": "insufficient_samples", "samples": len(rows)}
        rows.sort(key=lambda x: x[0])
        split = max(1, int(len(rows) * float(train_ratio)))
        train_rows = rows[:split]
        test_rows = rows[split:]
        if not test_rows:
            return {"ok": False, "reason": "insufficient_test_samples", "samples": len(rows)}

        # Reset in-memory model and train.
        self.weights = {}
        self.samples = 0
        self.wins = 0
        for _, row in train_rows:
            self.observe(row.get("features") or {}, won=bool(row.get("won", False)))

        total = 0
        correct = 0
        pnl = 0.0
        for _, row in test_rows:
            feats = row.get("features") or {}
            pred = self.predict(feats)
            pred_won = pred >= self.min_score
            actual_won = bool(row.get("won", False))
            if pred_won == actual_won:
                correct += 1
            total += 1
            pnl += float(row.get("pnl", 0.0))

        return {
            "ok": True,
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "directional_accuracy_pct": round((correct / total) * 100.0, 2) if total else 0.0,
            "test_pnl": round(pnl, 2),
            "status": self.status(),
        }
