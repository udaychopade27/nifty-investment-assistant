"""
Portfolio API Routes - COMPLETE IMPLEMENTATION
View holdings and performance with current market values
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List
from decimal import Decimal
import logging

from app.infrastructure.db.database import get_db
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)
router = APIRouter()


# Response models
class HoldingResponse(BaseModel):
    etf_symbol: str
    total_units: int
    total_invested: float
    average_price: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    pnl_percentage: float


class PortfolioSummaryResponse(BaseModel):
    total_invested: float
    current_value: float
    unrealized_pnl: float
    pnl_percentage: float
    holdings: List[HoldingResponse]


@router.get("/holdings", response_model=List[HoldingResponse])
async def get_holdings(db: AsyncSession = Depends(get_db)):
    """
    Get current portfolio holdings with live market values
    
    Returns ETF-wise units, invested amount, current value, PnL
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return []
        
        # Fetch current prices
        market_provider = YFinanceProvider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = {}
        
        for symbol in etf_symbols:
            price = await market_provider.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        holdings = []
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            invested = float(h['total_invested'])
            avg_price = float(h['average_price'])
            
            # Get current price
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            unrealized_pnl = current_value - invested
            pnl_pct = (unrealized_pnl / invested * 100) if invested > 0 else 0
            
            holdings.append(HoldingResponse(
                etf_symbol=symbol,
                total_units=units,
                total_invested=invested,
                average_price=avg_price,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=round(pnl_pct, 2)
            ))
        
        return holdings
        
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch holdings: {str(e)}"
        )


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(db: AsyncSession = Depends(get_db)):
    """
    Get complete portfolio summary with live values
    
    Total invested, current value, PnL, allocation
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return PortfolioSummaryResponse(
                total_invested=0.0,
                current_value=0.0,
                unrealized_pnl=0.0,
                pnl_percentage=0.0,
                holdings=[]
            )
        
        # Fetch current prices
        market_provider = YFinanceProvider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = {}
        
        for symbol in etf_symbols:
            price = await market_provider.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        # Calculate totals
        total_invested = 0.0
        total_current_value = 0.0
        holdings = []
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            invested = float(h['total_invested'])
            avg_price = float(h['average_price'])
            
            # Get current price
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            unrealized_pnl = current_value - invested
            pnl_pct = (unrealized_pnl / invested * 100) if invested > 0 else 0
            
            total_invested += invested
            total_current_value += current_value
            
            holdings.append(HoldingResponse(
                etf_symbol=symbol,
                total_units=units,
                total_invested=invested,
                average_price=avg_price,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=round(pnl_pct, 2)
            ))
        
        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return PortfolioSummaryResponse(
            total_invested=total_invested,
            current_value=total_current_value,
            unrealized_pnl=total_pnl,
            pnl_percentage=round(total_pnl_pct, 2),
            holdings=holdings
        )
        
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch summary: {str(e)}"
        )


@router.get("/allocation")
async def get_current_allocation(db: AsyncSession = Depends(get_db)):
    """
    Get current allocation vs target allocation
    
    Shows how your portfolio is allocated across ETFs
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return {
                "message": "No investments yet",
                "total_invested": 0,
                "allocation": []
            }
        
        # Fetch current prices for accurate allocation
        market_provider = YFinanceProvider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = {}
        
        for symbol in etf_symbols:
            price = await market_provider.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        # Calculate current allocation
        total_current_value = 0.0
        holdings_with_value = []
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            
            total_current_value += current_value
            holdings_with_value.append({
                'symbol': symbol,
                'invested': float(h['total_invested']),
                'current_value': current_value
            })
        
        allocation = []
        for h in holdings_with_value:
            invested_pct = (h['invested'] / sum(hh['invested'] for hh in holdings_with_value) * 100) if holdings_with_value else 0
            current_pct = (h['current_value'] / total_current_value * 100) if total_current_value > 0 else 0
            
            allocation.append({
                "etf_symbol": h['symbol'],
                "invested_amount": h['invested'],
                "invested_percentage": round(invested_pct, 2),
                "current_value": h['current_value'],
                "current_percentage": round(current_pct, 2)
            })
        
        # Sort by current value
        allocation.sort(key=lambda x: x['current_value'], reverse=True)
        
        return {
            "total_invested": sum(h['invested'] for h in holdings_with_value),
            "total_current_value": total_current_value,
            "allocation": allocation
        }
        
    except Exception as e:
        logger.error(f"Error fetching allocation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch allocation: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics(db: AsyncSession = Depends(get_db)):
    """
    Get performance metrics and statistics
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return {
                "message": "No investments yet",
                "metrics": {}
            }
        
        # Fetch current prices
        market_provider = YFinanceProvider()
        total_invested = sum(float(h['total_invested']) for h in holdings_data)
        
        current_prices = {}
        for h in holdings_data:
            price = await market_provider.get_current_price(h['etf_symbol'])
            if price:
                current_prices[h['etf_symbol']] = price
        
        # Calculate current value
        total_current_value = 0.0
        best_performer = None
        worst_performer = None
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            total_current_value += current_value
            
            avg_price = float(h['average_price'])
            return_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            if best_performer is None or return_pct > best_performer['return']:
                best_performer = {'etf': symbol, 'return': return_pct}
            
            if worst_performer is None or return_pct < worst_performer['return']:
                worst_performer = {'etf': symbol, 'return': return_pct}
        
        total_return = total_current_value - total_invested
        total_return_pct = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        return {
            "total_invested": total_invested,
            "current_value": total_current_value,
            "total_return": total_return,
            "return_percentage": round(total_return_pct, 2),
            "best_performer": best_performer,
            "worst_performer": worst_performer,
            "num_holdings": len(holdings_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch performance: {str(e)}"
        )