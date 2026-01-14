import logging
import requests

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

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

    if data in ("invest_base", "invest_tactical"):
        context.user_data.clear()
        context.user_data[INVEST_TYPE] = "BASE" if data == "invest_base" else "TACTICAL"
        context.user_data[INVEST_FLOW] = {}

        await update.callback_query.message.reply_text(
            f"*{context.user_data[INVEST_TYPE]} Investment*\n\nEnter ETF symbol:",
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

    # ----- Set capital -----
    if context.user_data.get(SET_CAPITAL_FLOW):
        context.user_data.clear()
        try:
            amount = float(text)
            resp = api_post("/capital/set", {"amount": amount})
            if resp.status_code != 200:
                raise ValueError
            await update.message.reply_text("✅ Monthly capital set.", parse_mode="Markdown")
        except Exception:
            await update.message.reply_text("❌ Enter a valid number.", parse_mode="Markdown")
        return

    # ----- Monthly report -----
    if context.user_data.get(MONTH_FLOW):
        context.user_data.clear()
        resp = api_get(f"/report/monthly/{text}")
        if resp.status_code == 200:
            d = resp.json()
            await update.message.reply_text(
                f"📊 *Monthly Summary — {d['month']}*\n\n"
                f"Planned: ₹{d['planned_capital']:.2f}\n"
                f"Base: ₹{d['capital_split']['base']:.2f}\n"
                f"Tactical: ₹{d['capital_split']['tactical']:.2f}",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("❌ Invalid month.")
        return

    # ----- Invest flow -----
    flow = context.user_data.get(INVEST_FLOW)
    invest_type = context.user_data.get(INVEST_TYPE)

    if flow is not None and invest_type:
        try:
            if "etf" not in flow:
                flow["etf"] = text
                await update.message.reply_text("Enter invested amount (₹):")
                return

            if "amount" not in flow:
                flow["amount"] = float(text)
                await update.message.reply_text("Enter execution price:")
                return

            flow["price"] = float(text)

            payload = {
                "execution_date": update.message.date.date().isoformat(),
                "etf_symbol": flow["etf"],
                "invested_amount": flow["amount"],
                "execution_price": flow["price"],
                "capital_type": invest_type,
            }

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
                return

            if resp.status_code == 409:
                err = resp.json()
                await update.message.reply_text(
                    f"❌ *Execution Blocked*\n\n"
                    f"{err['detail']}\n\n"
                    f"ℹ️ Tactical investments require a Daily Decision.\n"
                    f"Use /today first.",
                    parse_mode="Markdown",
                )
                return

            await update.message.reply_text("❌ Execution failed.")

        except Exception:
            logger.exception("Invest flow failed")
            await update.message.reply_text("❌ Execution failed.")


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
        "📊 *Monthly Summary*\n\nEnter month in `YYYY-MM` format:",
        parse_mode="Markdown",
    )
