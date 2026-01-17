import logging
import requests

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from app.services.etf_inav_service import ETFINAVService
from app.domain.strategy.etf_universe import ETF_UNIVERSE

API_BASE = "http://localhost:8000"
logger = logging.getLogger(__name__)

# ----------------------------
# API helpers (SYNC ONLY)
# ----------------------------

def api_get(path: str):
    return requests.get(f"{API_BASE}{path}", timeout=10)


def api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE}{path}", json=payload, timeout=10)


# ----------------------------
# State keys
# ----------------------------

INVEST_FLOW = "invest_flow"
INVEST_TYPE = "invest_type"      # BASE / TACTICAL
MONTH_FLOW = "month_flow"
SET_CAPITAL_FLOW = "set_capital_flow"

# ----------------------------
# ETF Selection (UI only)
# ----------------------------

ETF_BUTTONS = [
    "NIFTYBEES",
    "JUNIORBEES",
    "MIDCAPETF",
    "LOWVOLIETF",
    "GOLDBEES",
    "BHARATBOND",
]

# ----------------------------
# Core commands
# ----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📈 *Welcome to ETF Investing Assistant*\n\n"
        "This is *NOT* a trading bot.\n"
        "• No auto-investing\n"
        "• No predictions\n"
        "• Human-in-the-loop only\n\n"
        "Use /menu to get started.",
        parse_mode="Markdown",
    )


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📅 Today’s Decision", callback_data="today")],
        [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")],
        [InlineKeyboardButton("📊 Base Plan", callback_data="baseplan")],
        [InlineKeyboardButton("📈 PnL", callback_data="pnl")],
        [InlineKeyboardButton("📊 Monthly Summary", callback_data="month")],
        [InlineKeyboardButton("💰 Capital Status", callback_data="capital")],
        [InlineKeyboardButton("💸 Confirm Investment", callback_data="invest")],
        [InlineKeyboardButton("📊 iNAV Valuation", callback_data="inav_menu")],
        [InlineKeyboardButton("⚠️ Crash Advisory", callback_data="crash")],
        [InlineKeyboardButton("📜 Rules", callback_data="rules")],
        [InlineKeyboardButton("❓ Help", callback_data="help")],
    ]

    await update.message.reply_text(
        "📍 *Main Menu*",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ----------------------------
# Decision
# ----------------------------

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        resp = api_get("/decision/today")
        if resp.status_code != 200:
            await update.message.reply_text("❌ No decision available for today.")
            return

        d = resp.json()
        await update.message.reply_text(
            f"📅 *Today’s Decision*\n\n"
            f"Decision: *{d['decision_type']}*\n"
            f"Suggested Amount: ₹{d['suggested_amount']:.2f}\n"
            f"Deploy %: {d['deploy_pct'] * 100:.0f}%\n\n"
            f"🧠 *Why?*\n{d['explanation']}",
            parse_mode="Markdown",
        )
    except Exception:
        logger.exception("Today decision failed")
        await update.message.reply_text("❌ Unable to fetch today’s decision.")


# ----------------------------
# Portfolio
# ----------------------------

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        resp = api_get("/portfolio")
        if resp.status_code != 200:
            await update.message.reply_text("❌ Failed to load portfolio.")
            return

        data = resp.json()
        if not data["positions"]:
            await update.message.reply_text("📭 No investments yet.")
            return

        lines = ["💼 *Portfolio Snapshot*\n"]
        for p in data["positions"]:
            lines.append(
                f"*{p['etf_symbol']}*\n"
                f"Units: {p['units']:.2f}\n"
                f"Invested: ₹{p['invested_amount']:.2f}\n"
                f"Value: ₹{p['current_value']:.2f}\n"
                f"PnL: ₹{p['pnl']:.2f}\n"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception:
        logger.exception("Portfolio failed")
        await update.message.reply_text("❌ Portfolio unavailable.")


# ----------------------------
# PnL
# ----------------------------

async def pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        resp = api_get("/portfolio")
        if resp.status_code != 200:
            await update.message.reply_text("⚠️ Unable to compute PnL.")
            return

        d = resp.json()
        positions = d.get("positions", [])
        if not positions:
            await update.message.reply_text(
                "📭 *No investments yet*\n\nConfirm executions using /invest.",
                parse_mode="Markdown",
            )
            return

        total_invested = total_value = 0.0
        lines = ["📈 *Portfolio PnL*\n"]

        for p in positions:
            total_invested += p["invested_amount"]
            total_value += p["current_value"]

            lines.append(
                f"*{p['etf_symbol']}*\n"
                f"Units: {p['units']:.2f}\n"
                f"Avg Buy: ₹{p['avg_buy_price']:.2f}\n"
                f"Current: ₹{p['current_price']:.2f}\n"
                f"PnL: ₹{p['pnl']:.2f} ({p['pnl_pct']:.1f}%)\n"
            )

        total_pnl = total_value - total_invested
        pct = (total_pnl / total_invested) * 100 if total_invested else 0.0

        lines.append("────────────")
        lines.append(
            f"*Total Invested:* ₹{total_invested:.2f}\n"
            f"*Current Value:* ₹{total_value:.2f}\n"
            f"*Total PnL:* ₹{total_pnl:.2f} ({pct:.1f}%)"
        )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception:
        logger.exception("PnL failed")
        await update.message.reply_text("⚠️ Unable to compute PnL.")


# ----------------------------
# Capital
# ----------------------------

async def set_capital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data[SET_CAPITAL_FLOW] = True
    await update.message.reply_text(
        "💰 *Set Monthly Capital*\n\nEnter amount in INR.\n_Example:_ `100000`",
        parse_mode="Markdown",
    )


async def capital_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        resp = api_get("/capital/months/current")
        if resp.status_code != 200:
            await update.message.reply_text("❌ Capital not set for current month.")
            return

        d = resp.json()
        remaining = d["planned_capital"] - d["invested_till_today"]

        await update.message.reply_text(
            f"💰 *Capital Status*\n\n"
            f"📅 Month: `{d['month']}`\n"
            f"📊 Planned: ₹{d['planned_capital']:.2f}\n"
            f"💸 Invested: ₹{d['invested_till_today']:.2f}\n"
            f"📈 Remaining: ₹{remaining:.2f}",
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("Capital status failed")
        await update.message.reply_text("❌ Unable to fetch capital status.")


# ----------------------------
# Invest flow (BASE / TACTICAL)
# ----------------------------

async def invest_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[
        InlineKeyboardButton("📊 BASE (60%)", callback_data="invest_base"),
        InlineKeyboardButton("⚡ TACTICAL (40%)", callback_data="invest_tactical"),
    ]]

    await update.message.reply_text(
        "💸 *Confirm Investment*\n\nSelect investment type:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


# ----------------------------
# Callback router
# ----------------------------

async def handle_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    await update.callback_query.answer()
    
        # ---- iNAV menu ----
    if data == "inav_menu":
        keyboard = []
        row = []

        for symbol in ETF_UNIVERSE.keys():
            row.append(
                InlineKeyboardButton(
                    symbol,
                    callback_data=f"inav_etf:{symbol}"
                )
            )
            if len(row) == 2:
                keyboard.append(row)
                row = []

        if row:
            keyboard.append(row)

        await update.callback_query.message.reply_text(
            "📊 *ETF iNAV Lookup*\n\nSelect an ETF:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
        return

    # ---- iNAV ETF selected ----
    if data.startswith("inav_etf:"):
        etf = data.split(":", 1)[1]
        try:
            info = ETFINAVService.get_valuation(etf)

            await update.callback_query.message.reply_text(
                f"📊 *ETF Valuation — {etf}*\n\n"
                f"Market Price: ₹{info['market_price']}\n"
                f"iNAV: {info['inav'] if info['inav'] else 'Not Available'}\n"
                f"Gap: {info['gap_pct'] if info['gap_pct'] is not None else '—'}\n"
                f"Status: *{info['valuation']}*\n\n"
                f"_ℹ️ Informational only. Not used in decisions._",
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("INAV lookup failed")
            await update.callback_query.message.reply_text("❌ Unable to fetch iNAV.")
        return

    # ----- Invest type selection -----
    if data in ("invest_base", "invest_tactical"):
        context.user_data.clear()
        context.user_data[INVEST_TYPE] = "BASE" if data == "invest_base" else "TACTICAL"
        context.user_data[INVEST_FLOW] = {}

        keyboard = []
        row = []
        for etf in ETF_BUTTONS:
            row.append(
                InlineKeyboardButton(etf, callback_data=f"invest_etf_{etf}")
            )
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)

        await update.callback_query.message.reply_text(
            f"*{context.user_data[INVEST_TYPE]} Investment*\n\n"
            "Select ETF (or type manually):",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
        return

    # ----- ETF selected via button -----
    if data.startswith("invest_etf_"):
        etf = data.replace("invest_etf_", "")
        context.user_data.setdefault(INVEST_FLOW, {})["etf"] = etf

        await update.callback_query.message.reply_text(
            f"✅ ETF selected: *{etf}*\n\nEnter invested amount (₹):",
            parse_mode="Markdown",
        )
        return

    mapping = {
        "today": today,
        "portfolio": portfolio,
        "baseplan": base_plan,
        "pnl": pnl,
        "month": month_start,
        "capital": capital_status,
        "invest": invest_start,
        "crash": crash,
        "rules": rules,
        "help": help_cmd,
    }

    await mapping[data](update.callback_query, context)


# ----------------------------
# Text router
# ----------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # ----------------------------
    # 1. Monthly Summary Flow
    # ----------------------------
    if context.user_data.get(MONTH_FLOW):
        # Remove flag so next text isn't treated as a month
        context.user_data.pop(MONTH_FLOW, None)
        
        try:
            # Clean input: ensure "1" becomes "01" for the API
            month_num = text.zfill(2) 
            
            # API Call to /capital/{month_number}
            resp = api_get(f"/capital/{month_num}")
            
            if resp.status_code != 200:
                await update.message.reply_text(f"❌ No summary found for month: {text}")
                return

            d = resp.json()
            
            # Formatting response based on your specific API keys
            msg = (
                f"📊 *Summary: {d['month']}*\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 *Planned:* ₹{d['planned_capital']:.2f}\n"
                f"🏦 *Base:* ₹{d['base_capital']:.2f} (60%)\n"
                f"⚡ *Tactical:* ₹{d['tactical_capital']:.2f} (40%)\n"
                f"✅ *Invested:* ₹{d['invested_till_today']:.2f}\n"
                f"📉 *Remaining:* ₹{(d['planned_capital'] - d['invested_till_today']):.2f}"
            )
            
            await update.message.reply_text(msg, parse_mode="Markdown")

        except Exception:
            logger.exception("Monthly summary failed")
            await update.message.reply_text("❌ Error: Please enter a month number (e.g., 01 to 12).")
        return

    # ----------------------------
    # 2. Set Capital Flow
    # ----------------------------
    if context.user_data.get(SET_CAPITAL_FLOW):
        context.user_data.pop(SET_CAPITAL_FLOW, None)
        try:
            amount = float(text)
            resp = api_post("/capital/set", {"amount": amount})

            if resp.status_code == 200:
                await update.message.reply_text("✅ *Monthly capital set successfully.*", parse_mode="Markdown")
            elif resp.status_code == 409:
                err = resp.json()
                await update.message.reply_text(f"⚠️ *Capital Already Set*\n\n{err.get('detail', 'Exists for this month.')}", parse_mode="Markdown")
            elif resp.status_code == 400:
                err = resp.json()
                await update.message.reply_text(f"❌ *Invalid Input*\n\n{err.get('detail', 'Invalid amount.')}", parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Unable to set capital due to an unexpected error.")

        except ValueError:
            await update.message.reply_text("❌ Please enter a valid numeric amount (e.g. 100000).")
        except Exception:
            logger.exception("Set capital failed")
            await update.message.reply_text("❌ Internal error while setting capital.")
        return

    # ----------------------------
    # 3. Invest Flow (Base / Tactical)
    # ----------------------------
    flow = context.user_data.get(INVEST_FLOW)
    invest_type = context.user_data.get(INVEST_TYPE)

    if flow is not None and invest_type:
        try:
            # Step A: Get ETF Symbol
            if "etf" not in flow:
                flow["etf"] = text
                await update.message.reply_text("Enter invested amount (₹):")
                return

            # Step B: Get Amount
            if "amount" not in flow:
                flow["amount"] = float(text)
                await update.message.reply_text("Enter execution price:")
                return

            # Step C: Get Price and Confirm
            flow["price"] = float(text)

            payload = {
                "execution_date": update.message.date.date().isoformat(),
                "etf_symbol": flow["etf"],
                "invested_amount": flow["amount"],
                "execution_price": flow["price"],
                "capital_type": invest_type,
            }

            # Clear state before final API call
            context.user_data.clear()
            resp = api_post("/execution/confirm", payload)

            if resp.status_code == 200:
                r = resp.json()
                await update.message.reply_text(
                    f"✅ *Execution Confirmed*\n\n"
                    f"Type: *{invest_type}*\n"
                    f"ETF: {r['etf']}\n"
                    f"Units: {r['units']:.2f}",
                    parse_mode="Markdown",
                )
            elif resp.status_code == 409:
                err = resp.json()
                await update.message.reply_text(
                    f"❌ *Execution Blocked*\n\n{err['detail']}\n\n"
                    f"ℹ️ Tactical investments require a Daily Decision. Use /today first.",
                    parse_mode="Markdown",
                )
            else:
                await update.message.reply_text("❌ Execution failed.")

        except ValueError:
            await update.message.reply_text("❌ Please enter valid numbers for amount and price.")
        except Exception:
            logger.exception("Invest flow failed")
            await update.message.reply_text("❌ Internal error during execution.")
        return

# ----------------------------
# Misc
# ----------------------------

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – Onboarding\n"
        "/menu – Main menu\n"
        "/setcapital – Set monthly capital\n"
        "/capital – Capital status\n"
        "/baseplan – Base allocation plan\n"
        "/today – Daily decision\n"
        "/portfolio – Holdings\n"
        "/pnl – Profit & Loss\n"
        "/month – Monthly summary\n"
        "/invest – Confirm execution\n"
        "/inav – iNAV valuation\n\n"
        "/crash – Crash advisory\n"
        "/rules – Strategy rules\n\n"
        "⚠️ Investing involves market risk.",
        parse_mode="Markdown",
    )


async def rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📜 *Strategy Rules*\n\n"
        "• Monthly capital planning\n"
        "• 60% BASE / 40% TACTICAL\n"
        "• BASE = disciplined investing\n"
        "• TACTICAL = dip-based investing\n"
        "• Human-in-the-loop only\n"
        "• No predictions\n"
        "• No auto-execution",
        parse_mode="Markdown",
    )


async def crash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        resp = api_get("/crash/today")
        if resp.status_code != 200:
            await update.message.reply_text("✅ No crash advisory today.")
            return

        d = resp.json()
        await update.message.reply_text(
            f"⚠️ *Crash Advisory*\n\n"
            f"Severity: *{d['severity']}*\n"
            f"Suggested Extra Savings: {d['suggested_extra_savings_pct']}%\n\n"
            f"{d['reason']}\n\n"
            f"_Advisory only. No auto-action._",
            parse_mode="Markdown",
        )
    except Exception:
        logger.exception("Crash advisory failed")
        await update.message.reply_text("❌ Unable to fetch crash advisory.")


async def month_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data[MONTH_FLOW] = True
    await update.message.reply_text(
        "📊 *Monthly Summary*\n\nEnter month in `MM` format:",
        parse_mode="Markdown",
    )
# ----------------------------
# Base plan
# ----------------------------

async def base_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        month_resp = api_get("/capital/months/current")
        if month_resp.status_code != 200:
            await update.message.reply_text("❌ Monthly capital not set.")
            return

        month = month_resp.json()["month"]
        resp = api_get(f"/capital/base-plan/{month}")
        if resp.status_code != 200:
            await update.message.reply_text("❌ Base plan not available.")
            return

        d = resp.json()
        lines = [
            f"📊 *Base Investment Plan — {d['month']}*\n",
            f"*Total Base Capital:* ₹{d['base_capital']:.2f}\n",
        ]

        for p in d["base_plan"]:
            lines.append(
                f"• *{p['etf']}* → ₹{p['planned_amount']:.2f} "
                f"({p['allocation_pct']:.0f}%)"
            )

        lines.append(
            "\nℹ️ This is your *disciplined base plan*.\n"
            "💡 Remaining 40% is reserved for dip strategy."
        )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception:
        logger.exception("Base plan failed")
        await update.message.reply_text("❌ Unable to load base plan.")
        
# ----------------------------
# /inav COMMAND (UNCHANGED)
# ----------------------------

async def inav_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "Usage: /inav <ETF_SYMBOL>\nExample: /inav NIFTYBEES"
            )
            return

        etf = context.args[0].upper()
        info = ETFINAVService.get_valuation(etf)

        await update.message.reply_text(
            f"📊 *ETF Valuation — {etf}*\n\n"
            f"Market Price: ₹{info['market_price']}\n"
            f"iNAV: ₹{info['inav']}\n"
            f"Gap: {info['gap_pct']}%\n"
            f"Status: *{info['valuation']}*\n\n"
            f"_ℹ️ Informational only. Not used in decisions._",
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("iNAV command failed")
        await update.message.reply_text("❌ Unable to fetch iNAV.")