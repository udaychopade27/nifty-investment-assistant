# GitHub Copilot / AI Agent Instructions for ETF Assistant ✅

## Quick Overview
- Purpose: Deterministic, auditable **decision-quality** engine for disciplined ETF investing in India (NSE). See `README.md` for full context. 🔎
- Architecture: FastAPI API layer → Domain services (pure engines) → Infrastructure (DB, calendar, market data) → PostgreSQL audit ledger.
- Composition root: `app/main.py` (initializes ConfigEngine, engines, scheduler, Telegram bot).

---

## High-level rules (must follow)
- Domain *engines* must be pure: NO DB access, NO side effects, NO retries, and deterministic outputs. (See `app/domain/services/decision_engine.py` and `app/domain/services/config_engine.py`.)
- Persistence and orchestration happen in *services*, not engines (e.g., `DecisionService` uses `DecisionEngine`, then persists via repositories in `app/infrastructure/db/repositories`).
- Money/percentages always use `decimal.Decimal`. Use quantize explicitly when required.
- Domain types are immutable `@dataclass(frozen=True)` in `app/domain/models/entities.py`. Validate in `__post_init__` rather than allowing invalid objects.
- Insert-only DB: Tables are audit ledgers (no deletes). Model mapping lives in `app/infrastructure/db/models.py` and repository adapters under `app/infrastructure/db/repositories/*`.
- No automatic trade execution: human must trigger trades (Telegram `/invest` command and `/invest` API route). Never add auto-exec behavior.

---

## Where to look for common tasks (examples)
- Decision orchestration: `app/domain/services/decision_engine.py` (pure logic) and `app/domain/services/decision_service.py` (fetch market data, call engine, persist). ✅
- Config: `app/domain/services/config_engine.py` and YAML files in `config/` (`etfs.yml`, `allocations.yml`, `rules.yml`, `app.yml`). Fail-fast on missing/invalid config.
- Market data providers: `app/infrastructure/market_data/` (e.g., `yfinance_provider.py`). Follow provider interface used in services.
- Calendar: `app/infrastructure/calendar/nse_calendar.py` — use for trading-day validation.
- DB/repositories: `app/infrastructure/db/repositories/*` — repositories convert DB models → domain entities and vice versa.
- Scheduler/cron: `app/scheduler/main.py` — scheduled runs (daily at ~10:00 NSE) call the DecisionService.
- Telegram bot: `app/telegram/bot.py` — human interaction; commands defined there.
- Composition: `app/main.py` shows how components are wired. Use it as a reference when adding new services.

---

## Testing and dev workflow
- Run in Docker Compose: `docker-compose up -d` (service names are in `docker-compose.yml`).
- DB migrations: `docker-compose exec app alembic upgrade head`.
- Run tests: `docker-compose exec app pytest` or local `pytest` (project uses `pytest.ini`).
- Run a specific test file: `docker-compose exec app pytest tests/domain/services/test_decision_engine.py`.
- Add unit tests focusing on pure engines (fixtures + parametrized tests are common). See `tests/` for examples.

---

## Coding conventions & patterns to copy (explicit examples)
- Engines return domain models and primitives, never ORM models or side-effecting objects. Example: `DecisionEngine.generate_decision(...) -> (DailyDecision, List[ETFDecision])`.
- Services perform validation, logging, and persistence: `DecisionService.generate_decision_for_date(...)` shows fetching market data, computing context, calling the engine, then saving via `daily_decision_repo` and `etf_decision_repo`.
- Use clear, short, human-readable `explanation` strings for decisions (see `_generate_explanation` in `DecisionEngine`).
- Follow validation style: domain models raise ValueError on invalid inputs (see `ETF`, `ETFUnitPlan`, `MonthlyConfig` in `app/domain/models/entities.py`).
- Use `Decimal('0.01')` quantization for currency precision and allocation percentages.

---

## Safety & security constraints (non-negotiable)
- No endpoint or job should trigger automatic trading — explicit user confirmation required.
- Do not store secrets in code. Use `.env` and `app.yml` config; `.env.example` is provided.
- Fail-fast on config errors (no silent defaults).

---

## Useful file references (start here when exploring)
- Composition / startup: `app/main.py` 🔧
- Core orchestration: `app/domain/services/decision_engine.py` & `app/domain/services/decision_service.py` 🎯
- Config loader: `app/domain/services/config_engine.py` 📁
- Market provider: `app/infrastructure/market_data/yfinance_provider.py` 📈
- NSE Calendar: `app/infrastructure/calendar/nse_calendar.py` 📆
- Repositories: `app/infrastructure/db/repositories/*.py` 🗄️
- Domain models: `app/domain/models/entities.py` (immutable dataclasses) 🧾
- Scheduler: `app/scheduler/main.py` ⏰
- Tests: `tests/` (unit tests for engines & services) ✅

---

If anything here is unclear or you want more code snippets or rules (e.g., repository return shapes, specific quantization rules), tell me which section to expand and I will iterate. ✨
