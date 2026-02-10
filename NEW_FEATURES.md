# 🎉 NEW FEATURES GUIDE

## ✅ Upgrades Implemented

Your ETF Assistant now has **4 major upgrades**:

1. **Dynamic NSE Calendar** - Fetches holidays from NSE website
2. **Monthly Capital API** - Set capital via REST API
3. **Telegram Bot** - Complete bot with menu and commands
4. **Scheduler** - Auto-run daily decisions

---

## 1. 📅 Dynamic NSE Calendar

### What Changed
- ✅ Holidays now fetched from NSE website automatically
- ✅ Fallback to hardcoded holidays if fetch fails
- ✅ Caching for better performance

### How It Works
```python
from app.infrastructure.calendar.nse_calendar import NSECalendar

calendar = NSECalendar()
calendar.load_holidays()  # Fetches from NSE API

# Check trading day
is_trading = calendar.is_trading_day(date.today())

# Get holidays for current year
holidays = calendar.get_holidays()
```

### Test It
```bash
python -c "
from app.infrastructure.calendar.nse_calendar import NSECalendar
from datetime import date

cal = NSECalendar()
cal.load_holidays()

print(f'Today is trading day: {cal.is_trading_day(date.today())}')
print(f'Total holidays loaded: {len(cal.get_holidays())}')
"
```

---

## 2. 💰 Monthly Capital API

### Set Monthly Capital

**Endpoint:** `POST /api/v1/capital/set`

**Request:**
```json
{
  "monthly_capital": 50000,
  "month": "2026-02",
  "base_percentage": 60.0,
  "tactical_percentage": 40.0
}
```

**Response:**
```json
{
  "month": "2026-02",
  "monthly_capital": 50000.0,
  "base_capital": 30000.0,
  "tactical_capital": 20000.0,
  "trading_days": 19,
  "daily_tranche": 1578.95,
  "strategy_version": "2025-Q1",
  "created_at": "2026-01-30T10:00:00"
}
```

### Get Current Capital

**Endpoint:** `GET /api/v1/capital/current`

```bash
curl http://localhost:8000/api/v1/capital/current
```

### Get Specific Month

**Endpoint:** `GET /api/v1/capital/{month}`

```bash
curl http://localhost:8000/api/v1/capital/2026-02
```

### Using Python

```python
import requests

# Set capital for February 2026
response = requests.post('http://localhost:8000/api/v1/capital/set', json={
    "monthly_capital": 50000,
    "month": "2026-02"
})

print(response.json())

# Get current month
response = requests.get('http://localhost:8000/api/v1/capital/current')
print(response.json())
```

### Using curl

```bash
# Set capital
curl -X POST http://localhost:8000/api/v1/capital/set \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_capital": 50000,
    "month": "2026-02",
    "base_percentage": 60.0,
    "tactical_percentage": 40.0
  }'

# Get current
curl http://localhost:8000/api/v1/capital/current
```

---

## 3. 🤖 Telegram Bot

### Setup

1. **Get Bot Token from BotFather**
```
1. Open Telegram
2. Search for @BotFather
3. Send: /newbot
4. Follow instructions
5. Copy the token
```

2. **Add Token to .env**
```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_ENABLED=True
```

3. **Start Bot**
```bash
# Via Docker
docker-compose up telegram_bot

# Or directly
python -m app.telegram.bot
```

### Available Commands

```
/start          - Welcome message
/menu           - Main menu (recommended!)
/today          - Today's investment decision
/capital        - Monthly capital info
/portfolio      - View holdings
/month          - Monthly summary
/etfs           - ETF universe
/rules          - Investment rules
/help           - All commands
```

### Interactive Menu

The bot has an **inline keyboard menu** for easy navigation:

```
📊 Today's Decision | 💰 Set Capital
📈 Portfolio        | 📋 This Month
⚙️ ETF Universe     | 📖 Rules
        ❓ Help
```

### Example Interaction

**You:** `/start`

**Bot:**
```
🇮🇳 Welcome to ETF Assistant!

I help you invest systematically in Indian ETFs with discipline.

Quick Commands:
/menu - Main menu
/today - Today's decision
/capital - Set monthly capital
```

**You:** `/today`

**Bot:**
```
🟠 Decision for 2026-01-30

Type: MEDIUM
NIFTY Change: -2.30%

Investment:
💵 Suggested: ₹11,500.00
✅ Investable: ₹11,247.00
💸 Unused: ₹253.00

Capital Remaining:
📊 Base: ₹28,500.00
🎯 Tactical: ₹10,000.00

Explanation:
NIFTY: -2.30% | Medium dip. Deploying 50% tactical.
```

### Notifications

The bot can send **automatic notifications** when:
- Daily decision is generated
- Large market movements detected
- Monthly summary is ready

---

## 4. ⏰ Scheduler

### What It Does

Automatically runs:
- **Daily Decision** - 03:15 PM (Mon-Fri)
- **Monthly Plan** - 1st of month, 9:00 AM
- **Monthly Summary** - Last day of month, 6:00 PM

### Start Scheduler

**Via Docker:**
```bash
docker-compose up scheduler
```

**Directly:**
```bash
python -m app.scheduler.main
```

### Scheduler Output

```
🚀 Starting ETF Assistant Scheduler...
✅ Scheduler started successfully

📅 Scheduled Jobs:
  • Test Job (every 5 min) - Next run: 2026-01-30 10:05:00+05:30
  • Daily Decision Generation - Next run: 2026-01-31 10:00:00+05:30
  • Monthly Capital Plan - Next run: 2026-02-01 09:00:00+05:30
  • Monthly Summary Report - Next run: 2026-01-31 18:00:00+05:30

🎯 Scheduler is running. Press Ctrl+C to exit.
```

### Customize Schedule

Edit `app/scheduler/main.py`:

```python
# Change daily decision time
self.scheduler.add_job(
    self.daily_decision_job,
    CronTrigger(hour=9, minute=30, day_of_week='mon-fri'),  # 9:30 AM
    ...
)
```

---

## 🚀 Complete Workflow

### Initial Setup

```bash
# 1. Start all services
docker-compose up -d

# 2. Set monthly capital via API
curl -X POST http://localhost:8000/api/v1/capital/set \
  -H "Content-Type: application/json" \
  -d '{"monthly_capital": 50000, "month": "2026-02"}'

# 3. Start Telegram bot (in .env first)
docker-compose up telegram_bot

# 4. Send /start to your bot
```

### Daily Operation

```
03:15 PM → Scheduler generates decision
         ↓
10:05 AM → Telegram sends notification
         ↓
You      → Open Telegram, click /today
         → Review decision
         → Execute trades manually
         → (Optional) Confirm via /invest
```

---

## 🔧 Configuration

### Update .env

```bash
# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=your_chat_id  # Optional
TELEGRAM_ENABLED=True

# Scheduler
SCHEDULER_ENABLED=True
DAILY_DECISION_TIME=10:00
MONTHLY_SUMMARY_TIME=18:00

# Timezone
TIMEZONE=Asia/Kolkata
```

### Docker Compose

All services now run together:

```yaml
services:
  app:         # FastAPI
  db:          # PostgreSQL
  telegram_bot: # Telegram bot
  scheduler:   # Job scheduler
```

Start all:
```bash
docker-compose up -d
```

Or individually:
```bash
docker-compose up app          # Just API
docker-compose up telegram_bot # Just bot
docker-compose up scheduler    # Just scheduler
```

---

## 📊 API Documentation

Visit: `http://localhost:8000/docs`

New endpoints:
- `POST /api/v1/capital/set` - Set monthly capital
- `GET /api/v1/capital/current` - Get current capital
- `GET /api/v1/capital/{month}` - Get specific month

---

## 🧪 Testing New Features

### Test NSE Calendar
```bash
python -c "
from app.infrastructure.calendar.nse_calendar import NSECalendar
cal = NSECalendar()
cal.load_holidays([2026])
print('Holidays:', len(cal.get_holidays()))
"
```

### Test Capital API
```bash
# Set capital
curl -X POST http://localhost:8000/api/v1/capital/set \
  -H "Content-Type: application/json" \
  -d '{"monthly_capital": 50000}'

# Get it back
curl http://localhost:8000/api/v1/capital/current | jq
```

### Test Telegram Bot
```bash
# Start bot
python -m app.telegram.bot

# Send /start to your bot in Telegram
```

### Test Scheduler
```bash
# Start scheduler
python -m app.scheduler.main

# Watch logs for test job (every 5 min)
```

---

## 📝 Next Steps

1. ✅ Set your Telegram bot token
2. ✅ Set monthly capital via API
3. ✅ Start scheduler
4. ✅ Test bot commands
5. ✅ Wait for 03:15 PM for first decision!

---

## 🆘 Troubleshooting

### Telegram Bot Not Responding
```bash
# Check token
echo $TELEGRAM_BOT_TOKEN

# Check logs
docker-compose logs telegram_bot

# Restart
docker-compose restart telegram_bot
```

### Scheduler Not Running Jobs
```bash
# Check timezone
python -c "import pytz; print(pytz.timezone('Asia/Kolkata'))"

# Check logs
docker-compose logs scheduler

# Verify job schedule
# Jobs appear in logs on startup
```

### Capital API Errors
```bash
# Check database
docker-compose exec db psql -U etf_user -d etf_assistant

# Query: SELECT * FROM monthly_config;

# Check API health
curl http://localhost:8000/health
```

---

## 🎉 Summary

You now have:

✅ **Dynamic holidays** - No more manual updates  
✅ **Capital API** - Set capital via REST endpoints  
✅ **Telegram bot** - Interactive menu and commands  
✅ **Scheduler** - Automated daily decisions  

**Everything is production-ready for real investing!**

Start with:
1. `docker-compose up -d`
2. Set capital via API
3. Configure Telegram bot
4. Let scheduler run
5. Get notifications daily!

🚀 **Happy Automated Investing!**
