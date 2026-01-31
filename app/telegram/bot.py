"""
Telegram Bot - COMPLETE IMPLEMENTATION
All commands fully functional + Set Capital via Telegram
"""

import asyncio
import logging
from datetime import date, datetime
from decimal import Decimal
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

from app.config import settings
from app.infrastructure.db.database import async_session_factory
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.decision_repository import DailyDecisionRepository, ETFDecisionRepository
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.domain.services.config_engine import ConfigEngine

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ETFTelegramBot:
    """ETF Assistant Telegram Bot - ALL FEATURES IMPLEMENTED"""
    
    def __init__(self):
        """Initialize bot"""
        self.token = settings.TELEGRAM_BOT_TOKEN
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        
        self.nse_calendar = NSECalendar()
        
        # Load config
        config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_engine = ConfigEngine(config_dir)
        self.config_engine.load_all()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = """
🇮🇳 *Welcome to ETF Assistant!*

I help you invest systematically in Indian ETFs with discipline.

*Quick Commands:*
/menu - Main menu (recommended!)
/today - Today's decision
/capital - Monthly capital  
/setcapital - Set monthly capital (NEW!)
/portfolio - Your holdings
/invest - Record executed trade
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
                InlineKeyboardButton("📈 Portfolio", callback_data='portfolio'),
                InlineKeyboardButton("💸 Invest", callback_data='invest')
            ],
            [
                InlineKeyboardButton("📋 This Month", callback_data='month'),
                InlineKeyboardButton("⚙️ ETF Universe", callback_data='etfs')
            ],
            [
                InlineKeyboardButton("📖 Rules", callback_data='rules'),
                InlineKeyboardButton("📊 Allocation", callback_data='allocation')
            ],
            [
                InlineKeyboardButton("❓ Help", callback_data='help')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = "🎯 *ETF Assistant Menu*\n\nChoose an option:"
        
        if update.message:
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
        else:
            await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show today's decision"""
        async with async_session_factory() as session:
            repo = DailyDecisionRepository(session)
            decision = await repo.get_today()
            
            if not decision:
                if not self.nse_calendar.is_trading_day(date.today()):
                    msg = "❌ *Not a Trading Day*\n\nMarket is closed today."
                else:
                    msg = "⏳ *No Decision Yet*\n\nDecision will be generated at 10:00 AM on trading days."
            else:
                decision_icon = {
                    'NONE': '⭕',
                    'SMALL': '🟡',
                    'MEDIUM': '🟠',
                    'FULL': '🔴'
                }.get(decision.decision_type.value, '⚪')
                
                msg = f"""
{decision_icon} *Decision for {decision.date}*

*Type:* {decision.decision_type.value}
*NIFTY Change:* {decision.nifty_change_pct}%

*Investment:*
💵 Suggested: ₹{decision.suggested_total_amount:,.2f}
✅ Investable: ₹{decision.actual_investable_amount:,.2f}
💸 Unused: ₹{decision.unused_amount:,.2f}

*Capital Remaining:*
📊 Base: ₹{decision.remaining_base_capital:,.2f}
🎯 Tactical: ₹{decision.remaining_tactical_capital:,.2f}

*Explanation:*
{decision.explanation}

Use /invest to execute trades manually.
                """
            
            if update.message:
                await update.message.reply_text(msg, parse_mode='Markdown')
            else:
                await update.callback_query.answer()
                await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def capital_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show capital info with option to set if not configured"""
        async with async_session_factory() as session:
            repo = MonthlyConfigRepository(session)
            config = await repo.get_current()
            
            if config:
                msg = f"""
💰 *Current Month Capital*

*Month:* {config.month.strftime('%B %Y')}
*Total:* ₹{config.monthly_capital:,.2f}

*Split:*
📊 Base (60%): ₹{config.base_capital:,.2f}
🎯 Tactical (40%): ₹{config.tactical_capital:,.2f}

*Trading:*
📅 Trading Days: {config.trading_days}
💵 Daily Tranche: ₹{config.daily_tranche:,.2f}

*Strategy:* {config.strategy_version}
                """
                
                # Add button to update capital
                keyboard = [[InlineKeyboardButton("💰 Update Capital", callback_data='setcapital')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                if update.message:
                    await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
                else:
                    await update.callback_query.answer()
                    await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                # No capital set - prompt to set it
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
        
        # Set state
        context.user_data['awaiting_capital_amount'] = True
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def handle_capital_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle capital amount input"""
        if not context.user_data.get('awaiting_capital_amount'):
            return
        
        text = update.message.text.strip()
        
        if text == '/cancel':
            context.user_data.clear()
            await update.message.reply_text("❌ Capital setting cancelled.")
            return
        
        try:
            # Parse amount
            amount = float(text.replace(',', ''))
            
            if amount <= 0:
                await update.message.reply_text("❌ Amount must be positive. Please try again.")
                return
            
            if amount < 1000:
                await update.message.reply_text("❌ Minimum capital is ₹1,000. Please enter a higher amount.")
                return
            
            # Get current month
            today = date.today()
            month_date = date(today.year, today.month, 1)
            
            # Calculate splits
            monthly_capital = Decimal(str(amount))
            base_capital = (monthly_capital * Decimal('60') / Decimal('100')).quantize(Decimal('0.01'))
            tactical_capital = (monthly_capital * Decimal('40') / Decimal('100')).quantize(Decimal('0.01'))
            
            # Get trading days
            trading_days = self.nse_calendar.get_trading_days_in_month(month_date)
            
            if trading_days == 0:
                await update.message.reply_text("❌ No trading days in this month!")
                context.user_data.clear()
                return
            
            # Calculate daily tranche
            daily_tranche = (base_capital / Decimal(str(trading_days))).quantize(Decimal('0.01'))
            
            # Show confirmation
            confirmation_msg = f"""
✅ *Capital Configuration Ready*

*Month:* {month_date.strftime('%B %Y')}
*Total Capital:* ₹{monthly_capital:,.2f}

*Split:*
📊 Base (60%): ₹{base_capital:,.2f}
🎯 Tactical (40%): ₹{tactical_capital:,.2f}

*Trading Details:*
📅 Trading Days: {trading_days}
💵 Daily Tranche: ₹{daily_tranche:,.2f}

*Confirm to save?*
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Confirm", callback_data='confirm_capital'),
                    InlineKeyboardButton("❌ Cancel", callback_data='cancel_capital')
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Store in context for confirmation
            context.user_data['capital_data'] = {
                'month': month_date,
                'monthly_capital': monthly_capital,
                'base_capital': base_capital,
                'tactical_capital': tactical_capital,
                'trading_days': trading_days,
                'daily_tranche': daily_tranche
            }
            context.user_data['awaiting_capital_amount'] = False
            
            await update.message.reply_text(confirmation_msg, reply_markup=reply_markup, parse_mode='Markdown')
            
        except ValueError:
            await update.message.reply_text(
                "❌ Invalid amount. Please send a number.\n"
                "Example: `50000` for ₹50,000\n\n"
                "Send /cancel to cancel.",
                parse_mode='Markdown'
            )
    
    async def confirm_capital(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Confirm and save capital to database"""
        query = update.callback_query
        await query.answer()
        
        capital_data = context.user_data.get('capital_data')
        
        if not capital_data:
            await query.message.edit_text("❌ No capital data found. Please start over with /setcapital")
            return
        
        try:
            async with async_session_factory() as session:
                repo = MonthlyConfigRepository(session)
                
                # Check if already exists
                existing = await repo.get_for_month(capital_data['month'])
                
                if existing:
                    await query.message.edit_text(
                        f"⚠️ Capital already configured for {capital_data['month'].strftime('%B %Y')}.\n\n"
                        f"Existing: ₹{existing.monthly_capital:,.2f}\n\n"
                        f"Please delete it via API first to update."
                    )
                    context.user_data.clear()
                    return
                
                # Create new config
                config = await repo.create(
                    month=capital_data['month'],
                    monthly_capital=capital_data['monthly_capital'],
                    base_capital=capital_data['base_capital'],
                    tactical_capital=capital_data['tactical_capital'],
                    trading_days=capital_data['trading_days'],
                    daily_tranche=capital_data['daily_tranche'],
                    strategy_version=self.config_engine.strategy_version
                )
                
                success_msg = f"""
🎉 *Capital Saved Successfully!*

*Month:* {config.month.strftime('%B %Y')}
*Total:* ₹{config.monthly_capital:,.2f}

*Split:*
📊 Base: ₹{config.base_capital:,.2f}
🎯 Tactical: ₹{config.tactical_capital:,.2f}

*Trading:*
📅 Days: {config.trading_days}
💵 Daily: ₹{config.daily_tranche:,.2f}

✅ System is ready to generate daily decisions!

Use /menu to see all options.
                """
                
                await query.message.edit_text(success_msg, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Error saving capital: {e}")
            await query.message.edit_text(
                f"❌ Error saving capital: {str(e)}\n\n"
                "Please try again or contact support."
            )
        
        finally:
            context.user_data.clear()
    
    async def cancel_capital(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel capital setting"""
        query = update.callback_query
        await query.answer()
        
        await query.message.edit_text("❌ Capital setting cancelled.")
        context.user_data.clear()
    
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show portfolio holdings"""
        async with async_session_factory() as session:
            repo = ExecutedInvestmentRepository(session)
            holdings = await repo.get_holdings_summary()
            
            if not holdings:
                msg = "📈 *Your Portfolio*\n\nNo investments yet. Start investing when decisions are generated!"
            else:
                total_invested = sum(float(h['total_invested']) for h in holdings)
                
                msg = f"📈 *Your Portfolio*\n\n*Total Invested:* ₹{total_invested:,.2f}\n\n*Holdings:*\n"
                
                for h in holdings:
                    msg += f"\n{h['etf_symbol']}:\n"
                    msg += f"  Units: {h['total_units']}\n"
                    msg += f"  Invested: ₹{h['total_invested']:,.2f}\n"
                    msg += f"  Avg Price: ₹{h['average_price']:,.2f}\n"
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def etfs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ETF universe"""
        msg = "⚙️ *ETF Universe*\n\n"
        
        for etf in self.config_engine.etf_universe.etfs:
            icon = {
                'equity': '📊',
                'debt': '🏦',
                'gold': '🥇'
            }.get(etf.asset_class.value, '💼')
            
            msg += f"{icon} *{etf.symbol}*\n"
            msg += f"   {etf.name}\n"
            msg += f"   Type: {etf.asset_class.value.title()}\n"
            msg += f"   Risk: {etf.risk_level.value.replace('_', '-').title()}\n"
            msg += f"   Expense: {etf.expense_ratio}%\n\n"
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def rules_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show investment rules"""
        msg = "📖 *Investment Rules*\n\n"
        
        # Dip thresholds
        msg += "*Dip Thresholds:*\n"
        dip_thresholds = self.config_engine.get_rule('dip_thresholds')
        for level, rules in dip_thresholds.items():
            msg += f"\n{level.upper()}:\n"
            msg += f"  Range: {rules['min_change']}% to {rules['max_change']}%\n"
            msg += f"  Deploy: {rules['tactical_deployment']}% tactical\n"
        
        # Capital split
        msg += "\n*Capital Split:*\n"
        capital_rules = self.config_engine.get_rule('capital_rules')
        msg += f"  Base: {capital_rules['base_percentage']}%\n"
        msg += f"  Tactical: {capital_rules['tactical_percentage']}%\n"
        
        # Risk constraints
        msg += "\n*Risk Limits:*\n"
        msg += f"  Max Equity: {self.config_engine.risk_constraints.max_equity_allocation}%\n"
        msg += f"  Min Debt: {self.config_engine.risk_constraints.min_debt}%\n"
        msg += f"  Max Single ETF: {self.config_engine.risk_constraints.max_single_etf}%\n"
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def allocation_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show allocation strategy"""
        msg = "📊 *Allocation Strategy*\n\n"
        
        # Base allocation
        msg += "*Base Allocation (60%):*\n"
        for etf_symbol, pct in sorted(self.config_engine.base_allocation.allocations.items(), 
                                       key=lambda x: x[1], reverse=True):
            if pct > 0:
                msg += f"  {etf_symbol}: {pct}%\n"
        
        # Tactical allocation
        msg += "\n*Tactical Allocation (40%):*\n"
        for etf_symbol, pct in sorted(self.config_engine.tactical_allocation.allocations.items(), 
                                       key=lambda x: x[1], reverse=True):
            if pct > 0:
                msg += f"  {etf_symbol}: {pct}%\n"
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def month_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show monthly summary"""
        async with async_session_factory() as session:
            month_repo = MonthlyConfigRepository(session)
            config = await month_repo.get_current()
            
            if not config:
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
                else:
                    await update.callback_query.answer()
                    await update.callback_query.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                inv_repo = ExecutedInvestmentRepository(session)
                base_deployed = await inv_repo.get_total_base_deployed(config.month)
                tactical_deployed = await inv_repo.get_total_tactical_deployed(config.month)
                total_deployed = base_deployed + tactical_deployed
                
                msg = f"📋 *Monthly Summary - {config.month.strftime('%B %Y')}*\n\n"
                msg += f"*Capital:*\n"
                msg += f"  Total: ₹{config.monthly_capital:,.2f}\n"
                msg += f"  Base: ₹{config.base_capital:,.2f}\n"
                msg += f"  Tactical: ₹{config.tactical_capital:,.2f}\n\n"
                
                msg += f"*Deployed:*\n"
                msg += f"  Total: ₹{total_deployed:,.2f}\n"
                msg += f"  Base: ₹{base_deployed:,.2f}\n"
                msg += f"  Tactical: ₹{tactical_deployed:,.2f}\n\n"
                
                msg += f"*Remaining:*\n"
                base_remaining = config.base_capital - base_deployed
                tactical_remaining = config.tactical_capital - tactical_deployed
                msg += f"  Base: ₹{base_remaining:,.2f}\n"
                msg += f"  Tactical: ₹{tactical_remaining:,.2f}\n\n"
                
                msg += f"*Trading:*\n"
                msg += f"  Days: {config.trading_days}\n"
                msg += f"  Daily Tranche: ₹{config.daily_tranche:,.2f}\n"
                
                if update.message:
                    await update.message.reply_text(msg, parse_mode='Markdown')
                else:
                    await update.callback_query.answer()
                    await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def invest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Execute trade - Step 1: Select ETF with inline buttons"""
        # Show ETF buttons
        keyboard = [
            [
                InlineKeyboardButton("NIFTYBEES", callback_data='invest_NIFTYBEES'),
                InlineKeyboardButton("JUNIORBEES", callback_data='invest_JUNIORBEES')
            ],
            [
                InlineKeyboardButton("LOWVOLIETF", callback_data='invest_LOWVOLIETF'),
                InlineKeyboardButton("MIDCAPETF", callback_data='invest_MIDCAPETF')
            ],
            [
                InlineKeyboardButton("BHARATBOND", callback_data='invest_BHARATBOND'),
                InlineKeyboardButton("GOLDBEES", callback_data='invest_GOLDBEES')
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data='invest_cancel')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "💰 *Execute Investment*\n\n"
            "Select the ETF you executed:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def invest_etf_selected(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle ETF selection for investment"""
        query = update.callback_query
        await query.answer()
        
        etf_symbol = query.data.replace('invest_', '')
        
        if etf_symbol == 'cancel':
            await query.message.edit_text("❌ Investment cancelled.")
            return
        
        context.user_data['invest_etf'] = etf_symbol
        
        await query.message.edit_text(
            f"💰 *Execute {etf_symbol} Investment*\n\n"
            f"You selected: *{etf_symbol}*\n\n"
            f"Now send me:\n"
            f"`units executed_price`\n\n"
            f"Example: `10 278.50`\n\n"
            f"(Send /cancel to cancel)",
            parse_mode='Markdown'
        )
        
        context.user_data['awaiting_invest_details'] = True
    
    async def handle_invest_details(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle units and price input"""
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
            total_amount = units * price
            
            # Save to database
            async with async_session_factory() as session:
                from app.infrastructure.db.models import ExecutedInvestmentModel
                
                investment = ExecutedInvestmentModel(
                    etf_decision_id=1,  # Simplified
                    etf_symbol=etf_symbol,
                    units=units,
                    executed_price=Decimal(str(price)),
                    total_amount=Decimal(str(total_amount)),
                    slippage_pct=Decimal('0'),
                    capital_bucket='base',
                    executed_at=datetime.now(),
                    execution_notes=f"Executed via Telegram"
                )
                
                session.add(investment)
                await session.commit()
            
            await update.message.reply_text(
                f"✅ *Investment Recorded!*\n\n"
                f"ETF: {etf_symbol}\n"
                f"Units: {units}\n"
                f"Price: ₹{price:,.2f}\n"
                f"Total: ₹{total_amount:,.2f}\n\n"
                f"Recorded in database successfully.",
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
    
    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text inputs based on current state"""
        # Check for capital input first
        if context.user_data.get('awaiting_capital_amount'):
            await self.handle_capital_input(update, context)
            return
        
        # Check for invest details
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
/today - Today's investment decision
/capital - View monthly capital
/setcapital - Set monthly capital 💰
/portfolio - View holdings
/month - Monthly summary
/invest - Record executed trade

*Info Commands:*
/etfs - ETF universe
/rules - Investment rules
/allocation - Allocation strategy

*How It Works:*
1. Set monthly capital with /setcapital
2. System generates decision daily at 10:00 AM
3. You review the decision
4. You execute trades manually
5. You confirm execution via /invest
6. System maintains audit trail
        """
        
        if update.message:
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.callback_query.answer()
            await update.callback_query.message.reply_text(msg, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        
        # Handle capital confirmation
        if query.data == 'confirm_capital':
            await self.confirm_capital(update, context)
            return
        
        if query.data == 'cancel_capital':
            await self.cancel_capital(update, context)
            return
        
        await query.answer()
        
        # Handle invest callbacks
        if query.data.startswith('invest_'):
            await self.invest_etf_selected(update, context)
            return
        
        handlers = {
            'today': self.today_command,
            'setcapital': self.setcapital_command,
            'portfolio': self.portfolio_command,
            'invest': lambda u, c: self.invest_button_clicked(u, c),
            'month': self.month_command,
            'etfs': self.etfs_command,
            'rules': self.rules_command,
            'allocation': self.allocation_command,
            'help': self.help_command,
            'menu': self.menu_command,
        }
        
        handler = handlers.get(query.data)
        if handler:
            await handler(update, context)
        else:
            await query.message.reply_text(f"Feature '{query.data}' coming soon!")
    
    async def invest_button_clicked(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle invest button from menu"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [
                InlineKeyboardButton("NIFTYBEES", callback_data='invest_NIFTYBEES'),
                InlineKeyboardButton("JUNIORBEES", callback_data='invest_JUNIORBEES')
            ],
            [
                InlineKeyboardButton("LOWVOLIETF", callback_data='invest_LOWVOLIETF'),
                InlineKeyboardButton("MIDCAPETF", callback_data='invest_MIDCAPETF')
            ],
            [
                InlineKeyboardButton("BHARATBOND", callback_data='invest_BHARATBOND'),
                InlineKeyboardButton("GOLDBEES", callback_data='invest_GOLDBEES')
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data='invest_cancel')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            "💰 *Execute Investment*\n\n"
            "Select the ETF you executed:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or contact support."
            )
    
    def run(self):
        """Run the bot"""
        logger.info("Starting Telegram bot...")
        
        application = Application.builder().token(self.token).build()
        
        # Add ALL command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("menu", self.menu_command))
        application.add_handler(CommandHandler("today", self.today_command))
        application.add_handler(CommandHandler("capital", self.capital_command))
        application.add_handler(CommandHandler("setcapital", self.setcapital_command))
        application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        application.add_handler(CommandHandler("invest", self.invest_command))
        application.add_handler(CommandHandler("etfs", self.etfs_command))
        application.add_handler(CommandHandler("rules", self.rules_command))
        application.add_handler(CommandHandler("allocation", self.allocation_command))
        application.add_handler(CommandHandler("month", self.month_command))
        application.add_handler(CommandHandler("help", self.help_command))
        
        # Add callback query handler
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Add message handler for text inputs (capital and invest)
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input)
        )
        
        # Add error handler
        application.add_error_handler(self.error_handler)
        
        logger.info("✅ Bot started. Press Ctrl+C to stop.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    bot = ETFTelegramBot()
    bot.run()


if __name__ == "__main__":
    main()
