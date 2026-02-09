"""Rule-based + optional LLM confidence scoring (non-trade-generating)."""
from __future__ import annotations

from typing import Dict, Any, Optional
import json
import httpx

from app.config import settings


def rule_based_confidence(signal: Dict[str, Any], indicator: Dict[str, Any]) -> float:
    score = 0.45
    signal_type = signal.get("signal")
    ema_fast = indicator.get("ema_fast")
    ema_slow = indicator.get("ema_slow")
    vwap = indicator.get("vwap")
    close = indicator.get("close")
    atr = indicator.get("atr")
    oi_change = indicator.get("oi_change")
    volume_spike = indicator.get("volume_spike")
    pcr = indicator.get("pcr")
    rsi = indicator.get("rsi")
    macd_hist = indicator.get("macd_hist")
    boll_pos = indicator.get("boll_pos")

    if ema_fast is not None and ema_slow is not None:
        if signal_type == "BUY_CE":
            score += 0.14 if ema_fast > ema_slow else -0.14
        if signal_type == "BUY_PE":
            score += 0.14 if ema_fast < ema_slow else -0.14
    if vwap is not None and close is not None:
        if signal_type == "BUY_CE":
            score += 0.08 if close >= vwap else -0.08
        if signal_type == "BUY_PE":
            score += 0.08 if close <= vwap else -0.08
    if atr is None:
        score -= 0.12
    else:
        score += 0.08
    if oi_change is None:
        score -= 0.05
    else:
        if signal_type == "BUY_CE":
            score += 0.08 if oi_change > 0 else -0.08
        if signal_type == "BUY_PE":
            score += 0.08 if oi_change < 0 else -0.08
    if volume_spike is True:
        score += 0.08
    elif volume_spike is False:
        score -= 0.08
    if pcr is not None:
        if signal_type == "BUY_PE" and pcr < 1.0:
            score += 0.05
        elif signal_type == "BUY_CE" and pcr > 1.0:
            score += 0.05
        else:
            score -= 0.03
    if rsi is not None:
        if signal_type == "BUY_CE" and 52 <= float(rsi) <= 72:
            score += 0.05
        elif signal_type == "BUY_PE" and 28 <= float(rsi) <= 48:
            score += 0.05
        else:
            score -= 0.03
    if macd_hist is not None:
        if signal_type == "BUY_CE":
            score += 0.04 if float(macd_hist) > 0 else -0.04
        if signal_type == "BUY_PE":
            score += 0.04 if float(macd_hist) < 0 else -0.04
    if boll_pos is not None:
        if signal_type == "BUY_CE" and float(boll_pos) > 0.55:
            score += 0.03
        elif signal_type == "BUY_PE" and float(boll_pos) < 0.45:
            score += 0.03
    return max(0.0, min(1.0, round(score, 3)))


async def llm_adjust_confidence(signal: Dict[str, Any], indicator: Dict[str, Any]) -> Optional[float]:
    provider = (settings.LLM_PROVIDER or "none").lower()
    if provider == "none":
        return None

    payload = {
        "signal": signal,
        "indicator": indicator,
    }
    prompt = (
        "You are an intraday options assistant."
        " Return JSON only: {\"confidence_adjust\": <number between -0.2 and 0.2>, \"note\": <short string>}"
        f"\nContext: {json.dumps(payload, default=str)}"
    )

    if provider == "local":
        url = f"{settings.LLM_BASE_URL.rstrip('/')}/api/generate"
        body = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body)
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = data.get("response", "")
        return _extract_adjust(text)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": settings.LLM_MODEL,
            "input": prompt,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = _extract_response_text(data)
        return _extract_adjust(text)

    return None


def _extract_response_text(payload: Dict[str, Any]) -> str:
    # Responses API output text (best-effort)
    output = payload.get("output") or []
    for item in output:
        content = item.get("content") or []
        for part in content:
            if part.get("type") == "output_text":
                return part.get("text", "")
    return ""


def _extract_adjust(text: str) -> Optional[float]:
    try:
        data = json.loads(text)
        adj = float(data.get("confidence_adjust"))
        return max(-0.2, min(0.2, adj))
    except Exception:
        return None
