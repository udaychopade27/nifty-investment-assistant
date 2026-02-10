# NIFTY/BANKNIFTY Options Operating System

This project now supports your rule-based manual alert workflow for:
- Intraday option buying (CE/PE)
- Confidence-score gating
- Monthly-capital-based risk controls

## Module 2: Daily Checklist

### Pre-market (before 09:15 IST)
- Global markets neutral/supportive
- No major event day (RBI/CPI/Fed)
- Previous day high/low marked
- Major OI zones noted

### Intraday (before each trade)
- VWAP direction clear on 5m structure
- Futures OI confirming direction
- ATM OI is shifting (not flat)
- IV not collapsing
- Bid-ask spread acceptable

If any item fails: `NO TRADE`.

### Post-trade
- Stop-loss respected
- Emotional state stable
- Trade logged

## Module 3: Confidence Score Calculator

The runtime applies a 0-100 score and blocks low-quality alerts.
Scoring is strict: if required data is missing (IV/futures OI/etc), missing factors get 0.

Weights:
- VWAP alignment: 25
- ATM OI behavior: 25
- Futures confirmation: 20
- IV direction: 15
- Time of day: 10
- Event risk: 5

Rule:
- Score >= 70: eligible
- Score < 70: blocked

Config location:
- `config/options/options.yml` -> `options.project.confidence`

## Monthly Capital Risk Engine

Config location:
- `config/options/options.yml` -> `options.project.capital`

Used for:
- Dynamic `risk_per_trade`
- Daily max-loss cap
- Max trades/day cap

Default profile (editable):
- Monthly capital: 15,000
- Risk/trade: 8% (capped at 10%)
- Daily max loss: 12%
- Weekly max loss: 25%
- Max trades/day: 2

## Strict Gate Rules (No Fallback)

Signals are blocked when any required input is missing:
- 5m confirmation (close vs VWAP)
- ATM CE/PE OI shift deltas
- IV change
- Bid-ask spread data
- Option delta

Futures OI gate is optional:
- `options.project.strict_requirements.require_futures_oi_confirmation: false` means options-only mode.
- Set to `true` only if you want futures OI confirmation.

Additional hard gates:
- Live premium availability from option chain (no fixed premium bands)
- Delta range (0.40-0.55 abs), reject below 0.30 abs
- Event-day blocker (configurable dates)

Data config:
- `config/options/options.yml` -> `options.market_data.futures_instruments`
- `config/options/options.yml` -> `options.project.strict_requirements`
- `config/options/options.yml` -> `options.project.event_calendar.major_event_dates`

## Runtime Integration

Implemented in:
- `app/domain/options/runtime.py`
- `app/domain/options/analytics/confidence_score.py`

Output alerts now include:
- Monthly capital
- Risk per trade (Rs)
- SL/Target percent and Rupee values
- Confidence score (/100)
