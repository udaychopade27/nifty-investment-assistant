# 🇮🇳 Indian ETF Investing Assistant

A production-grade, SaaS-ready ETF investing assistant designed specifically for the Indian stock market (NSE).

## 🎯 What This System Is

This is a **disciplined, rule-based investing system** with:
- ✅ Human-in-the-loop execution (NO auto-trading)
- ✅ Deterministic decision engine
- ✅ Capital safety and full auditability
- ✅ NSE trading calendar awareness
- ✅ Indian market constraints (whole units only, no fractional ETFs)

## ❌ What This System Is NOT

- ❌ NOT a trading bot
- ❌ NO auto-execution of trades
- ❌ NO market predictions or ML models
- ❌ NO fractional ETF units
- ❌ NO forced buying

## 🏗️ Architecture

```
API (FastAPI routes only)
   ↓
Domain Services (pure logic)
   ↓
Infrastructure (DB, Calendar, Market Data)
   ↓
PostgreSQL (audit ledger)
```

### Clean Architecture Principles
- **API Layer**: Orchestration only
- **Domain Layer**: Business rules & math
- **Infrastructure Layer**: External systems
- **No circular dependencies**
- **No shortcut imports**

## 📁 Project Structure

```
etf_assistant/
├── app/
│   ├── main.py                  # Composition root
│   ├── api/
│   │   └── routes/              # FastAPI endpoints
│   ├── domain/
│   │   ├── services/            # Core engines
│   │   └── models/              # Domain entities
│   ├── infrastructure/
│   │   ├── db/                  # PostgreSQL
│   │   ├── calendar/            # NSE trading days
│   │   └── market_data/         # Price fetching
│   ├── config/
│   │   ├── app.yml
│   │   ├── etfs.yml
│   │   ├── allocations.yml
│   │   └── rules.yml
│   ├── scheduler/               # Daily/monthly jobs
│   ├── telegram/                # Bot client
│   └── common/                  # Shared utilities
├── tests/
├── docker/
└── .env.example
```

## 🎛️ Core Engines

1. **Config Engine** - Load and validate YAML configuration
2. **Market Context Engine** - Calculate market stress (no predictions)
3. **Capital Engine** - Track base/tactical/extra buckets
4. **Allocation Engine** - ETF-wise capital distribution
5. **Unit Calculation Engine** - Convert ₹ to whole ETF units
6. **Decision Engine** - Daily orchestration (CORE)
7. **Crash Opportunity Engine** - Advisory signals only
8. **Execution Validation Engine** - Validate manual trades
9. **Portfolio Engine** - Holdings ledger
10. **Analytics Engine** - Read-only insights

## 💰 Capital Model

### Monthly Capital Allocation
- **Base Capital (60%)**: Gradual, systematic investing
- **Tactical Capital (40%)**: Deploy on market dips
- **Extra Capital**: Optional crash-opportunity injections

### Capital Flow
```
User sets monthly capital (e.g., ₹50,000)
    ↓
Base: ₹30,000 (60%) → Daily tranches
Tactical: ₹20,000 (40%) → Dip deployment
```

### Unused Tactical Capital
- Rolls forward to next month
- Capped at 150% of one month's capital

## 📊 ETF Universe (Default)

| ETF | Category | Purpose |
|-----|----------|---------|
| NIFTYBEES | Large-cap | Core equity |
| JUNIORBEES | Growth | Next 50 companies |
| LOWVOLIETF | Defensive | Low volatility |
| BHARATBOND | Bonds | Debt allocation |
| GOLDBEES | Gold | Hedge |
| MIDCAPETF | Growth | Mid-cap exposure |

*Extensible via `etfs.yml`*

## 🎯 Decision Rules

### Base Investing (60% Capital)
- Invested gradually over NSE trading days
- Daily tranche = Monthly Base / Trading Days
- Always invested (market up or down)

### Tactical Dip Deployment (40% Capital)

| Market Fall | Action |
|-------------|--------|
| ≥ -1% | NONE |
| -1% to -2% | Deploy 25% tactical |
| -2% to -3% | Deploy 50% tactical |
| < -3% | Deploy 100% tactical |
| 3-day ≥ 2.5% | MEDIUM override |

### ETF-Wise Dip Allocation

| ETF | % |
|-----|---|
| NIFTYBEES | 45% |
| JUNIORBEES | 25% |
| LOWVOLIETF | 20% |
| MIDCAPETF | 10% |
| BHARATBOND | 0% |
| GOLDBEES | 0% |

## 🧮 Unit-Based Planning (India-Critical)

**Amounts are NOT executable. Units are.**

```python
effective_price = ltp × 1.02  # 2% buffer
units = floor(allocated_amount / effective_price)

if units >= 1:
    status = PLANNED
else:
    status = SKIPPED  # No redistribution!
```

**Rules:**
- ✅ Whole units only
- ✅ Floor() function, no rounding
- ❌ No fractional units
- ❌ No forced redistribution

## 🔄 Daily Workflow

1. **Market Data Fetch** (9:30 AM NSE)
2. **Decision Engine Run** (10:00 AM)
   - Calculate market context
   - Determine dip level
   - Allocate capital to ETFs
   - Convert to units
   - Persist DailyDecision
3. **Telegram Notification** (10:05 AM)
4. **Human Execution** (Anytime)
   - User reviews decision
   - Manually executes trades
   - Confirms via `/invest` command

## 🤖 Telegram Commands

```
/start           - Welcome message
/menu            - Main menu
/today           - Today's investment decision
/invest          - Execute trade manually
/portfolio       - View current holdings
/allocation      - Check current allocation
/month           - Monthly summary
/etfs            - ETF universe
/rules           - Investment rules
/help            - Command help
```

## 🗄️ Database Schema

### Insert-Only Tables (Audit Ledger)
- `monthly_config` - Capital settings per month
- `daily_decision` - Every day's decision (even NONE)
- `etf_decision` - ETF-wise plans
- `executed_investment` - Manual executions
- `extra_capital_injection` - Crash-opportunity adds
- `monthly_summary` - Month-end rollups
- `trading_holiday` - NSE calendar

**No deletes. No silent updates.**

## 🚀 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Docker & Docker Compose

### Setup

1. **Clone & Navigate**
```bash
cd etf_assistant
```

2. **Environment Variables**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Docker Compose**
```bash
docker-compose up -d
```

4. **Database Migration**
```bash
docker-compose exec app alembic upgrade head
```

5. **Start Services**
```bash
docker-compose restart
```

## 🔧 Configuration

### `config/etfs.yml`
Define your ETF universe

### `config/allocations.yml`
Set base and tactical allocations

### `config/rules.yml`
Configure dip thresholds and constraints

### `config/app.yml`
System-wide settings

## 📅 Scheduler Jobs

### Daily (10:00 AM NSE)
- Run decision engine
- Send Telegram notification

### Monthly (1st trading day)
- Create new MonthlyConfig
- Rollover unused tactical capital
- Generate monthly summary

## 🧪 Testing

```bash
# Run all tests
docker-compose exec app pytest

# Run with coverage
docker-compose exec app pytest --cov=app

# Run specific engine tests
docker-compose exec app pytest tests/domain/services/test_decision_engine.py
```

## 📊 Example Decision Flow

**Scenario: NIFTY falls -2.3%, Monthly Capital = ₹50,000**

1. **Market Context**: Stress = MEDIUM
2. **Capital Deployment**: 
   - Base: ₹1,000 (daily tranche)
   - Tactical: ₹10,000 (50% tactical)
   - Total: ₹11,000
3. **Allocation**:
   - NIFTYBEES: ₹4,950 (45%)
   - JUNIORBEES: ₹2,750 (25%)
   - LOWVOLIETF: ₹2,200 (20%)
   - MIDCAPETF: ₹1,100 (10%)
4. **Unit Calculation**:
   - NIFTYBEES: LTP ₹278 → 17 units → ₹4,726
   - JUNIORBEES: LTP ₹585 → 4 units → ₹2,340
   - LOWVOLIETF: LTP ₹57 → 38 units → ₹2,166
   - MIDCAPETF: LTP ₹145 → 7 units → ₹1,015
5. **Actual Total**: ₹10,247
6. **Unused**: ₹753 (carries forward)

## 🔒 Safety Guarantees

- ✅ **No Auto-Trading**: Every execution requires human confirmation
- ✅ **Deterministic**: Same inputs → Same outputs
- ✅ **Auditable**: Full ledger of all decisions and executions
- ✅ **Capital Safety**: Strict bucket separation
- ✅ **Indian Market Compliant**: Whole units, NSE calendar, realistic prices

## 📈 Analytics Dashboard

Access via `/portfolio` or web dashboard:
- Current holdings
- Total invested
- Current value
- Unrealized PnL
- Allocation drift
- Monthly contribution

## 🛡️ Risk Constraints

```yaml
max_equity_allocation: 75%
max_single_etf: 45%
max_midcap: 10%
max_gold: 15%
min_debt: 10%
```

## 🔄 Monthly Cycle

1. **Month Start**: Create new `MonthlyConfig`
2. **Daily**: Run decision engine on NSE trading days
3. **Manual**: Execute trades via Telegram
4. **Month End**: Generate summary, rollover capital

## 📞 Support

For issues or questions:
1. Check `/help` in Telegram bot
2. Review `logs/app.log`
3. Check database audit trail

## 🎓 Philosophy

> "This is a decision-quality system, not a trading app. It suggests what is sensible today, then waits for the human to act."

## 📜 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Designed for disciplined, long-term ETF investing in the Indian stock market.

---

**Version**: 1.0.0  
**Strategy Version**: 2025-Q1  
**Last Updated**: January 2026
