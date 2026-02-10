from app.domain.options.analytics.confidence_score import calculate_confidence_score


def test_confidence_score_high_for_aligned_ce_signal():
    score, parts = calculate_confidence_score(
        signal_type="BUY_CE",
        indicator={
            "close": 22010.0,
            "vwap": 21980.0,
            "oi_change": 12000.0,
            "ema_fast": 22005.0,
            "ema_slow": 21970.0,
            "iv_change": 0.6,
            "ts": "2026-02-10T10:30:00+05:30",
        },
    )
    assert score >= 70
    assert parts["vwap_alignment"] > 0


def test_confidence_score_low_for_misaligned_pe_signal():
    score, parts = calculate_confidence_score(
        signal_type="BUY_PE",
        indicator={
            "close": 22010.0,
            "vwap": 21980.0,
            "oi_change": 12000.0,
            "ema_fast": 22005.0,
            "ema_slow": 21970.0,
            "iv_change": -0.8,
            "ts": "2026-02-10T14:20:00+05:30",
        },
    )
    assert score < 70
    assert parts["time_of_day"] == 0


def test_confidence_score_no_fallback_when_iv_and_futures_missing():
    score, parts = calculate_confidence_score(
        signal_type="BUY_CE",
        indicator={
            "close": 22010.0,
            "vwap": 21980.0,
            "oi_change": 12000.0,
            "ts": "2026-02-10T10:30:00+05:30",
        },
    )
    assert parts["futures_confirmation"] == 0
    assert parts["iv_direction"] == 0
    assert score < 70
