"""
Telegram Bot - Pure UX Layer (API Client Only)
No direct database access, no business logic - just API calls
"""

import asyncio
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
import httpx

from app.config import settings

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ETFTelegramBot:
    """ETF Assistant Telegram Bot - Pure UX Layer"""
    
    def __init__(self):
        """Initialize bot"""
        self.token = settings.TELEGRAM_BOT_TOKEN
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        
        # API base URL (internal Docker network)
        self.api_base = "http://app:8000"
        
        self.application = None
    
    async def api_get(self, endpoint: str):
        """Make GET request to API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base}{endpoint}", timeout=30.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"API GET {endpoint} failed: {e}")
            raise
    
    async def api_post(self, endpoint: str, data: dict):
        """Make POST request to API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}{endpoint}",
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API POST {endpoint} failed: {e}")
            raise
        except Exception as e:
            logger.error(f"API POST {endpoint} error: {e}")
            raise
    
    async def start_async(self):
        """Start Telegram bot in async mode (for FastAPI)"""
        logger.info("🤖 Starting Telegram bot (async mode for FastAPI)...")
    
        self.application = Application.builder().token(self.token).build()
    
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("menu", self.menu_command))
        self.application.add_handler(CommandHandler("today", self.today_command))
        self.application.add_handler(CommandHandler("capital", self.capital_command))
        self.application.add_handler(CommandHandler("setcapital", self.setcapital_command))
        self.application.add_handler(CommandHandler("baseplan", self.baseplan_command))  # ✅ NEW
        self.application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        self.application.add_handler(CommandHandler("invest", self.invest_command))
        self.application.add_handler(CommandHandler("etfs", self.etfs_command))
        self.application.add_handler(CommandHandler("rules", self.rules_command))
        self.application.add_handler(CommandHandler("allocation", self.allocation_command))
        self.application.add_handler(CommandHandler("month", self.month_command))
        self.application.add_handler(CommandHandler("tradingstatus", self.trading_status_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Add callback and message handlers
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input)
        )
        self.application.add_error_handler(self.error_handler)
        
        # Initialize and start
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
        logger.info("✅ Telegram bot started successfully (async mode)")
        
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("🛑 Telegram bot task cancelled, shutting down...")
            await self.stop_async()
            raise
    
    async def stop_async(self):
        """Stop Telegram bot gracefully"""
        if self.application:
            logger.info("🛑 Stopping Telegram bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("✅ Telegram bot stopped")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = """
🇮🇳 *Welcome to ETF Assistant!*

I help you invest systematically in Indian ETFs with discipline.

*Capital Model:*
📊 Base (60%) - Systematic, any day
⚡ Tactical (40%) - Signal-driven only

*Quick Commands:*
/menu - Main menu (recommended!)
/setcapital - Set monthly capital
/baseplan - See base investment plan 📋
/today - Today's tactical decision
/invest - Record executed trade
/portfolio - Your holdings
/help - All commands

Let's build wealth together! 📈
    """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    
    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show main menu"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Today's Decision", callback_data='today'),
                InlineKeyboardButton("💰 Set Capital", callback_data='setcapital')
            ],
            [
                InlineKeyboardButton("📋 Base Plan", callback_data='baseplan'),  # ✅ NEW
                InlineKeyboardButton("💸 Invest", callback_data='invest')
            ],
            [
                InlineKeyboardButton("📈 Portfolio", callback_data='portfolio'),
                InlineKeyboardButton("📅 This Month", callback_data='month')
            ],
            [
                InlineKeyboardButton("⚙️ ETF Universe", callback_data='etfs'),
                InlineKeyboardButton("📖 Rules", callback_data='rules')
            ],
            [
                InlineKeyboardButton("📊 Allocation", callback_data='allocation'),
                InlineKeyboardButton("❓ Help", callback_data='help')
            ],
            [
                InlineKeyboardButton("🚦 Trading Status", callback_data='tradingstatus')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
    
        msg = "🎯 *ETF Assistant Menu*\n\nChoose an option:"
    
        if update.message:
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show today's decision - API call"""
        try:
            decision = await self.api_get("/api/v1/decision/today")
        
            decision_icon = {
                'NONE': '⭕',
                'SMALL': '🟡',
                'MEDIUM': '🟠',
                'FULL': '🔴'
            }.get(decision.get('decision_type', 'NONE'), '⚪')
        
            msg = f"""
    {decision_icon} *Decision for {decision.get('date')}*

    *Type:* {decision.get('decision_type')}
    *NIFTY Change:* {decision.get('nifty_change_pct')}%

    *Investment:*
    💵 Suggested: ₹{decision.get('suggested_total_amount', 0):,.2f}
    ✅ Investable: ₹{decision.get('actual_investable_amount', 0):,.2f}

    *Capital Remaining:*
    📊 Base: ₹{decision.get('remaining_base_capital', 0):,.2f}
    🎯 Tactical: ₹{decision.get('remaining_tactical_capital', 0):,.2f}

    Use /invest to record trades.
            """

            etf_decisions = decision.get('etf_decisions', [])
            if etf_decisions:
                etf_msg = "*📌 Tactical ETF Suggestions:*\n"
                for etf in etf_decisions:
                    etf_msg += (
                        f"\n*{etf.get('etf_symbol')}*\n"
                        f"  🔢 Units: {etf.get('units')}\n"
                        f"  💰 Price: ₹{etf.get('effective_price')}\n"
                        f"  ✅ Amount: ₹{etf.get('actual_amount')}\n"
                    )
                msg = msg + "\n\n" + etf_msg
        
            if update.message:
                await update.message.reply_text(msg, parse_mode='Markdown')
            else:
                await update.callback_query.answer("Loading...")
                await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            msg = "⏳ *No Decision Yet*\n\nDecision will be generated at 3:15 PM on trading days."
        
            if update.message:
                await update.message.reply_text(msg, parse_mode='Markdown')
            else:
                await update.callback_query.answer("No decision")
                await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def capital_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show capital info - API call"""
        try:
            config = await self.api_get("/api/v1/capital/current")
            
            msg = f"""
💰 *Current Month Capital*

*Month:* {config.get('month')}
*Total:* ₹{config.get('monthly_capital', 0):,.2f}

*Split:*
📊 Base (60%): ₹{config.get('base_capital', 0):,.2f}
🎯 Tactical (40%): ₹{config.get('tactical_capital', 0):,.2f}

*Trading:*
📅 Trading Days: {config.get('trading_days')}
💵 Daily Tranche: ₹{config.get('daily_tranche', 0):,.2f}

*Strategy:* {config.get('strategy_version')}
            """
            
            keyboard = [[InlineKeyboardButton("💰 Update Capital", callback_data='setcapital')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        except:
            msg = """
💰 *Set Monthly Capital*

No capital configured for this month yet!

*Let's set it up now!*

Use /setcapital or click the button below.
            """
            keyboard = [[InlineKeyboardButton("💰 Set Capital Now", callback_data='setcapital')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.message:
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def baseplan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show base investment plan - API call"""
        try:
            # If invoked via button, callback is already answered in button_callback.
            if update.callback_query:
                await update.callback_query.message.reply_text(
                    "⏳ Generating base plan..."
                )

            # Call API to generate base plan
            plan = await self.api_post("/api/v1/capital/generate-base-plan", {})
        
            # First message - Summary
            summary_msg = f"""
📋 *Base Investment Plan*

*Month:* {plan.get('month')}
*Base Capital:* ₹{plan.get('base_capital', 0):,.2f}
*Total Allocated:* ₹{plan.get('total_allocated', 0):,.2f}
*Total Investable:* ₹{plan.get('total_actual', 0):,.2f}
*Unused:* ₹{plan.get('total_unused', 0):,.2f}

_Loading ETF details..._
        """
        
            if update.message:
                await update.message.reply_text(summary_msg, parse_mode='Markdown')
            else:
                await update.callback_query.message.reply_text(summary_msg, parse_mode='Markdown')
        
        # Second message - ETF-wise breakdown (split if too long)
            base_plan = plan.get('base_plan', {})
        
            etf_messages = []
            current_msg = "*📊 ETF-wise Recommendations:*\n"
        
            for etf_symbol, details in sorted(base_plan.items(), 
                                            key=lambda x: x[1].get('allocated_amount', 0), 
                                            reverse=True):
                if details.get('status') == 'price_unavailable':
                    etf_msg = f"\n*{etf_symbol}:* ❌ Price unavailable\n"
                else:
                    allocation_pct = details.get('allocation_pct', 0)
                    allocated = details.get('allocated_amount', 0)
                    units = details.get('recommended_units', 0)
                    ltp = details.get('ltp', 0)
                    effective_price = details.get('effective_price', 0)
                    actual_amount = details.get('actual_amount', 0)
                    unused = details.get('unused', 0)
                
                    etf_msg = f"""
    *{etf_symbol}* ({allocation_pct}%)
    💵 Allocated: ₹{allocated:,.2f}
    📊 LTP: ₹{ltp:,.2f}
    💰 Buy: ₹{effective_price:,.2f}
    🔢 Units: {units}
    ✅ Invest: ₹{actual_amount:,.2f}
                """
            
            # Check if adding this would exceed Telegram's limit (4096 chars)
                if len(current_msg) + len(etf_msg) > 3800:  # Safe margin
                    etf_messages.append(current_msg)
                    current_msg = etf_msg
                else:
                    current_msg += etf_msg
        
            if current_msg:
                etf_messages.append(current_msg)
        
        # Send ETF details (may be multiple messages)
            for msg in etf_messages:
                if update.message:
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
        
        # Third message - Instructions
            instructions_msg = f"""
*📝 How to Execute:*

✅ Execute gradually over the month
✅ Can invest on any trading day
✅ Use /invest → Base → Select ETF
✅ Enter units and executed price

    *Note:* {plan.get('note', 'Base investments are systematic and can be done anytime.')}
            """
        
            if update.message:
                await update.message.reply_text(instructions_msg, parse_mode='Markdown')
            else:
                await update.callback_query.message.reply_text(instructions_msg, parse_mode='Markdown')
        
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get('detail', 'Unable to generate plan')
            except:
                error_detail = 'Unable to generate plan'
        
            msg = f"""
    ❌ *Cannot Generate Base Plan*

    {error_detail}

*Common reasons:*
- No capital set for current month
- Market data unavailable
- System error

Use /setcapital first.
        """
        
            if update.message:
                await update.message.reply_text(msg, parse_mode='Markdown')
            else:
                await update.callback_query.answer("Error!")
                await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in baseplan_command: {e}")
            msg = f"❌ Error: {str(e)}\n\nPlease try again."
        
            if update.message:
                await update.message.reply_text(msg)
            else:
                await update.callback_query.answer("Error!")
                await update.callback_query.message.reply_text(msg)
            
    async def setcapital_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start capital setting flow"""
        msg = """
💰 *Set Monthly Capital*

Let me help you set your monthly investment capital.

*Please send me the monthly amount in ₹*

Examples:
- `50000` for ₹50,000
- `100000` for ₹1,00,000
- `25000` for ₹25,000

The system will automatically:
✅ Split 60% Base + 40% Tactical
✅ Calculate trading days
✅ Compute daily tranche

Send /cancel to cancel.
        """
        
        context.user_data['awaiting_capital_amount'] = True
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def handle_capital_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle capital amount input - API call"""
        if not context.user_data.get('awaiting_capital_amount'):
            return
        
        text = update.message.text.strip()
        
        if text == '/cancel':
            context.user_data.clear()
            await update.message.reply_text("❌ Capital setting cancelled.")
            return
        
        try:
            amount = float(text.replace(',', ''))
            
            if amount <= 0:
                await update.message.reply_text("❌ Amount must be positive. Please try again.")
                return
            
            if amount < 1000:
                await update.message.reply_text("❌ Minimum capital is ₹1,000. Please enter a higher amount.")
                return
            
            # Call API to set capital
            result = await self.api_post("/api/v1/capital/set", {
                "monthly_capital": amount
            })
            
            success_msg = f"""
🎉 *Capital Saved Successfully!*

*Month:* {result.get('month')}
*Total:* ₹{result.get('monthly_capital', 0):,.2f}

*Split:*
📊 Base: ₹{result.get('base_capital', 0):,.2f}
🎯 Tactical: ₹{result.get('tactical_capital', 0):,.2f}

*Trading:*
📅 Days: {result.get('trading_days')}
💵 Daily: ₹{result.get('daily_tranche', 0):,.2f}

✅ System is ready to generate daily decisions!

Use /menu to see all options.
            """
            
            await update.message.reply_text(success_msg, parse_mode='Markdown')
            context.user_data.clear()
            
        except httpx.HTTPStatusError as e:
            error_msg = e.response.json().get('detail', str(e))
            await update.message.reply_text(f"❌ Error: {error_msg}\n\nPlease try again.")
        except ValueError:
            await update.message.reply_text(
                "❌ Invalid amount. Please send a number.\n"
                "Example: `50000` for ₹50,000\n\n"
                "Send /cancel to cancel.",
                parse_mode='Markdown'
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error saving capital: {str(e)}")
            context.user_data.clear()
    
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show portfolio holdings - API call"""
        try:
            portfolio = await self.api_get("/api/v1/portfolio/summary")
            
            msg = f"""
📈 *Your Portfolio*

*Total Invested:* ₹{portfolio.get('total_invested', 0):,.2f}
*Current Value:* ₹{portfolio.get('current_value', 0):,.2f}
*Unrealized P&L:* ₹{portfolio.get('unrealized_pnl', 0):,.2f}
*P&L %:* {portfolio.get('pnl_percentage', 0):.2f}%

Use /menu for more options.
            """
        except:
            msg = "📈 *Your Portfolio*\n\nNo investments yet. Start investing when decisions are generated!"
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def etfs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ETF universe - API call"""
        try:
            etfs_payload = await self.api_get("/api/v1/config/etfs")
            if isinstance(etfs_payload, list):
                etfs_list = etfs_payload
            elif isinstance(etfs_payload, dict):
                etfs_list = etfs_payload.get('etfs', [])
            else:
                etfs_list = []
            
            msg = "⚙️ *ETF Universe*\n\n"
            
            for etf in etfs_list:
                icon = {
                    'equity': '📊',
                    'debt': '🏦',
                    'gold': '🥇'
                }.get(etf.get('asset_class'), '💼')
                
                msg += f"{icon} *{etf.get('symbol')}*\n"
                msg += f"   {etf.get('name')}\n"
                msg += f"   Type: {etf.get('asset_class', '').title()}\n"
                msg += f"   Risk: {etf.get('risk_level', '').replace('_', '-').title()}\n"
                msg += f"   Expense: {etf.get('expense_ratio')}%\n\n"
            
            if not etfs_list:
                msg += "No ETFs configured yet. Update `config/etfs.yml` and try again."
        except:
            msg = "⚙️ *ETF Universe*\n\nUnable to load ETFs. Please try again."
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def rules_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show investment rules - API call"""
        try:
            rules = await self.api_get("/api/v1/config/rules")
            
            msg = "📖 *Investment Rules*\n\n"
            
            msg += "*Dip Thresholds:*\n"
            for level, threshold in rules.get('dip_thresholds', {}).items():
                msg += f"\n{level.upper()}:\n"
                msg += f"  Range: {threshold.get('min_change')}% to {threshold.get('max_change')}%\n"
                msg += f"  Deploy: {threshold.get('tactical_deployment')}% tactical\n"
            
            msg += "\n*Capital Split:*\n"
            capital = rules.get('capital_rules', {})
            msg += f"  Base: {capital.get('base_percentage')}%\n"
            msg += f"  Tactical: {capital.get('tactical_percentage')}%\n"
        except:
            msg = "📖 *Investment Rules*\n\nUnable to load rules. Please try again."
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def allocation_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show allocation strategy - API call"""
        try:
            base_alloc = await self.api_get("/api/v1/config/allocations/base")
            tactical_alloc = await self.api_get("/api/v1/config/allocations/tactical")
            
            msg = "📊 *Allocation Strategy*\n\n"
            
            msg += "*Base Allocation (60%):*\n"
            for symbol, pct in sorted(base_alloc.get('allocations', {}).items(), 
                                     key=lambda x: x[1], reverse=True):
                if pct > 0:
                    msg += f"  {symbol}: {pct}%\n"
            
            msg += "\n*Tactical Allocation (40%):*\n"
            for symbol, pct in sorted(tactical_alloc.get('allocations', {}).items(), 
                                     key=lambda x: x[1], reverse=True):
                if pct > 0:
                    msg += f"  {symbol}: {pct}%\n"
        except:
            msg = "📊 *Allocation Strategy*\n\nUnable to load allocation. Please try again."
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')

    async def trading_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show trading status flags - API call"""
        try:
            status = await self.api_get("/api/v1/config/trading")

            msg = (
                "🚦 *Trading Status*\n\n"
                f"*Trading Enabled:* {status.get('trading_enabled')}\n"
                f"*Base Enabled:* {status.get('trading_base_enabled')}\n"
                f"*Tactical Enabled:* {status.get('trading_tactical_enabled')}\n"
                f"*Simulation Only:* {status.get('simulation_only')}\n"
            )
        except Exception:
            msg = "🚦 *Trading Status*\n\nUnable to load status. Please try again."

        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def month_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show monthly summary - API call"""
        try:
            config = await self.api_get("/api/v1/capital/current")
            
            # Get deployed amounts (you'll need to add this endpoint or use existing)
            msg = f"""
📋 *Monthly Summary - {config.get('month')}*

*Capital:*
  Total: ₹{config.get('monthly_capital', 0):,.2f}
  Base: ₹{config.get('base_capital', 0):,.2f}
  Tactical: ₹{config.get('tactical_capital', 0):,.2f}

*Trading:*
  Days: {config.get('trading_days')}
  Daily Tranche: ₹{config.get('daily_tranche', 0):,.2f}
            """
        except:
            msg = """
❌ *No Monthly Configuration*

No capital set for this month yet!

*Let's set it up!*

Use /setcapital to configure monthly capital.
            """
            
            keyboard = [[InlineKeyboardButton("💰 Set Capital Now", callback_data='setcapital')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if update.message:
                await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
                return
            else:
                await update.callback_query.answer()
                await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
                return
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def invest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Execute trade - Step 1: Choose Base or Tactical"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Base Investment", callback_data='invest_type_base'),
                InlineKeyboardButton("⚡ Tactical Investment", callback_data='invest_type_tactical')
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data='invest_cancel')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = """
💰 *Record Investment*

*First, choose investment type:*

📊 *Base Investment*
   • Systematic, SIP-like
   • Can invest any day
   • Uses base capital (60%)

⚡ *Tactical Investment*
   • Signal-driven only
   • Requires today's decision
   • Uses tactical capital (40%)

*Which type?*
        """
        
        await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def invest_type_selected(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle Base/Tactical selection - Step 2: Choose ETF"""
        query = update.callback_query
        await query.answer()
        
        invest_type = query.data.replace('invest_type_', '')
        
        if invest_type == 'cancel':
            await query.message.edit_text("❌ Investment cancelled.")
            context.user_data.clear()
            return
        
        context.user_data['invest_bucket'] = invest_type
        
        # If tactical, check via API
        if invest_type == 'tactical':
            try:
                allowed = await self.api_get("/api/v1/invest/today/allowed")
                
                if not allowed.get('tactical', {}).get('allowed'):
                    reason = allowed.get('tactical', {}).get('reason', 'Not allowed')
                    await query.message.edit_text(
                        f"❌ *Tactical Investment Blocked*\n\n{reason}\n\n"
                        "You can still execute *Base* investments.",
                        parse_mode='Markdown'
                    )
                    context.user_data.clear()
                    return
            except:
                pass  # Continue anyway
        
        # Show ETF selection (dynamic from API)
        try:
            etfs_payload = await self.api_get("/api/v1/config/etfs")
            if isinstance(etfs_payload, list):
                etfs_list = etfs_payload
            elif isinstance(etfs_payload, dict):
                etfs_list = etfs_payload.get('etfs', [])
            else:
                etfs_list = []
        except Exception:
            etfs_list = []
        
        if not etfs_list:
            await query.message.edit_text(
                "⚠️ *No ETFs Available*\n\n"
                "Please check `config/etfs.yml` and try again.",
                parse_mode='Markdown'
            )
            context.user_data.clear()
            return
        
        buttons = []
        row = []
        for etf in etfs_list:
            symbol = etf.get('symbol')
            if not symbol:
                continue
            row.append(InlineKeyboardButton(symbol, callback_data=f"invest_etf_{symbol}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        
        if row:
            buttons.append(row)
        
        buttons.append([InlineKeyboardButton("❌ Cancel", callback_data='invest_cancel')])
        reply_markup = InlineKeyboardMarkup(buttons)
        
        type_icon = "📊" if invest_type == 'base' else "⚡"
        
        await query.message.edit_text(
            f"{type_icon} *{invest_type.upper()} Investment*\n\n"
            f"Select the ETF you executed:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def invest_etf_selected(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle ETF selection - Step 3: Ask for units and price"""
        query = update.callback_query
        await query.answer()
        
        etf_symbol = query.data.replace('invest_etf_', '')
        
        if etf_symbol == 'cancel':
            await query.message.edit_text("❌ Investment cancelled.")
            context.user_data.clear()
            return
        
        context.user_data['invest_etf'] = etf_symbol
        
        invest_type = context.user_data.get('invest_bucket', 'base')
        type_icon = "📊" if invest_type == 'base' else "⚡"
        
        await query.message.edit_text(
            f"{type_icon} *{invest_type.upper()} Investment*\n"
            f"*ETF:* {etf_symbol}\n\n"
            f"Now send me:\n"
            f"`units executed_price`\n\n"
            f"*Examples:*\n"
            f"• `10 278.50` → 10 units @ ₹278.50\n"
            f"• `5 584.25` → 5 units @ ₹584.25\n"
            f"• `20 145.80` → 20 units @ ₹145.80\n\n"
            f"(Send /cancel to cancel)",
            parse_mode='Markdown'
        )
        
        context.user_data['awaiting_invest_details'] = True
    
    async def handle_invest_details(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle units and price input - API call to save"""
        if not context.user_data.get('awaiting_invest_details'):
            return
        
        text = update.message.text.strip()
        
        if text == '/cancel':
            context.user_data.clear()
            await update.message.reply_text("❌ Investment cancelled.")
            return
        
        try:
            parts = text.split()
            if len(parts) != 2:
                await update.message.reply_text(
                    "❌ Invalid format. Send: `units price`\n"
                    "Example: `10 278.50`",
                    parse_mode='Markdown'
                )
                return
            
            units = int(parts[0])
            price = float(parts[1])
            
            if units <= 0 or price <= 0:
                await update.message.reply_text("❌ Units and price must be positive")
                return
            
            etf_symbol = context.user_data.get('invest_etf')
            bucket = context.user_data.get('invest_bucket', 'base')
            
            if not etf_symbol:
                await update.message.reply_text("❌ Missing ETF symbol. Please start again with /invest")
                context.user_data.clear()
                return
            
            total_amount = units * price
            
            # Call API to save investment
            result = await self.api_post(
                f"/api/v1/invest/{bucket}",
                {
                    "etf_symbol": etf_symbol,
                    "units": units,
                    "executed_price": price,
                    "notes": f"Executed via Telegram ({bucket})"
                }
            )
            
            # Success!
            type_icon = "📊" if bucket == 'base' else "⚡"
            
            await update.message.reply_text(
                f"✅ *Investment Recorded Successfully!*\n\n"
                f"*Type:* {type_icon} {bucket.upper()}\n"
                f"*ETF:* {etf_symbol}\n"
                f"*Units:* {units}\n"
                f"*Price:* ₹{price:,.2f}\n"
                f"*Total:* ₹{total_amount:,.2f}\n\n"
                f"✅ Saved to database with {bucket} capital bucket.\n\n"
                f"Use /portfolio to view all holdings.",
                parse_mode='Markdown'
            )
            
            context.user_data.clear()
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get('detail', str(e))
            await update.message.reply_text(
                f"❌ *API Error*\n\n"
                f"{error_detail}\n\n"
                f"Please check the error and try again.",
                parse_mode='Markdown'
            )
            context.user_data.clear()
        except ValueError as e:
            await update.message.reply_text(
                f"❌ Invalid input: {e}\n"
                f"Send: `units price`\n"
                f"Example: `10 278.50`",
                parse_mode='Markdown'
            )
        except Exception as e:
            await update.message.reply_text(
                f"❌ *Failed to save investment*\n\n"
                f"Error: {str(e)}\n\n"
                f"The API may be unreachable. Please check logs.",
                parse_mode='Markdown'
            )
            context.user_data.clear()
    
    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Route text inputs to appropriate handlers"""
        if context.user_data.get('awaiting_capital_amount'):
            await self.handle_capital_input(update, context)
            return
        
        if context.user_data.get('awaiting_invest_details'):
            await self.handle_invest_details(update, context)
            return
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help"""
        msg = """
    📖 *ETF Assistant Commands*

*Main Commands:*
/start - Welcome message
/menu - Main menu (recommended!)
/today - Today's decision
/capital - View monthly capital
/setcapital - Set monthly capital 💰
/baseplan - Generate base investment plan 📋
/portfolio - View holdings
/month - Monthly summary
/invest - Record executed trade

*Info Commands:*
/etfs - ETF universe
/rules - Investment rules
/allocation - Allocation strategy
/tradingstatus - Trading status flags

*Investment Flow:*
1. Set monthly capital with /setcapital
2. View base plan with /baseplan (shows units to buy)
3. System generates tactical decisions daily at 3:15 PM
4. Review tactical decision with /today
5. Execute trades manually in your broker
6. Record execution via /invest
7. System maintains complete audit trail

*Capital Model:*
📊 Base (60%) - Systematic, any day
   • Use /baseplan to see recommendations
   • Execute gradually over the month
⚡ Tactical (40%) - Signal-driven only
   • Check /today for dip-based signals
        """
    
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query

    # ✅ ALWAYS answer immediately (first line)
        try:
            await query.answer()
        except Exception:
            pass  # prevents "already answered" edge cases

    # ---- INVEST FLOW ----
        if query.data.startswith('invest_type_'):
            await self.invest_type_selected(update, context)
            return

        if query.data.startswith('invest_etf_'):
            await self.invest_etf_selected(update, context)
            return

        if query.data == 'invest_cancel':
            await query.message.edit_text("❌ Investment cancelled.")
            context.user_data.clear()
            return

    # ---- MENU HANDLERS ----
        handlers = {
            'today': self.today_command,
            'setcapital': self.setcapital_command,
            'baseplan': self.baseplan_command,
            'portfolio': self.portfolio_command,
            'invest': lambda u, c: self.invest_button_clicked(u, c),
            'month': self.month_command,
            'etfs': self.etfs_command,
            'rules': self.rules_command,
            'allocation': self.allocation_command,
            'tradingstatus': self.trading_status_command,
            'help': self.help_command,
            'menu': self.menu_command,
        }

        handler = handlers.get(query.data)
        if handler:
            await handler(update, context)

    
    async def invest_button_clicked(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle invest button from menu"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Base Investment", callback_data='invest_type_base'),
                InlineKeyboardButton("⚡ Tactical Investment", callback_data='invest_type_tactical')
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data='invest_cancel')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = """
💰 *Record Investment*

*Choose investment type:*

📊 *Base Investment*
   • Systematic, SIP-like
   • Can invest any day
   • Uses base capital (60%)

⚡ *Tactical Investment*
   • Signal-driven only
   • Requires today's decision
   • Uses tactical capital (40%)
        """
        
        await query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or contact support."
            )
    
    def run(self):
        """Run the bot in standalone mode (blocking)"""
        logger.info("🤖 Starting Telegram bot (standalone mode)...")
    
        application = Application.builder().token(self.token).build()
    
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("menu", self.menu_command))
        application.add_handler(CommandHandler("today", self.today_command))
        application.add_handler(CommandHandler("capital", self.capital_command))
        application.add_handler(CommandHandler("setcapital", self.setcapital_command))
        application.add_handler(CommandHandler("baseplan", self.baseplan_command))  # ✅ NEW
        application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        application.add_handler(CommandHandler("invest", self.invest_command))
        application.add_handler(CommandHandler("etfs", self.etfs_command))
        application.add_handler(CommandHandler("rules", self.rules_command))
        application.add_handler(CommandHandler("allocation", self.allocation_command))
        application.add_handler(CommandHandler("month", self.month_command))
        application.add_handler(CommandHandler("tradingstatus", self.trading_status_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input))
        application.add_error_handler(self.error_handler)
        
        logger.info("✅ Bot started (standalone). Press Ctrl+C to stop.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point for standalone execution"""
    bot = ETFTelegramBot()
    bot.run()


if __name__ == "__main__":
    main()
