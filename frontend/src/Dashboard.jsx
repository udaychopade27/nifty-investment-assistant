import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Wallet, PieChart, Calendar, 
  DollarSign, Activity, ChevronRight, AlertCircle, CheckCircle,
  ArrowUpRight, ArrowDownRight, Minus, Plus, BarChart3, 
  RefreshCw, Bell, Settings, LogOut, Menu, X, Download,
  Target, Shield, Zap, Clock, Info, ExternalLink
} from 'lucide-react';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Utility Components
const Card = ({ children, className = '', ...props }) => (
  <div
    className={`bg-[var(--card)] rounded-2xl shadow-sm border border-[var(--line)] ${className}`}
    {...props}
  >
    {children}
  </div>
);

const Button = ({ children, variant = 'primary', size = 'md', onClick, disabled, className = '' }) => {
  const variants = {
    primary: 'bg-primary-600 hover:bg-primary-700 text-white',
    secondary: 'bg-white/70 hover:bg-white text-ink-0 border border-[var(--line)]',
    success: 'bg-emerald-600 hover:bg-emerald-700 text-white',
    danger: 'bg-rose-600 hover:bg-rose-700 text-white',
    outline: 'border-2 border-primary-600 text-primary-700 hover:bg-primary-50'
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg'
  };
  
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        ${variants[variant]} ${sizes[size]}
        rounded-lg font-medium transition-all duration-200
        disabled:opacity-50 disabled:cursor-not-allowed
        ${className}
      `}
    >
      {children}
    </button>
  );
};

const Badge = ({ children, variant = 'default' }) => {
  const variants = {
    default: 'bg-stone-100 text-stone-800',
    success: 'bg-emerald-100 text-emerald-800',
    warning: 'bg-amber-100 text-amber-900',
    danger: 'bg-rose-100 text-rose-800',
    info: 'bg-sky-100 text-sky-800'
  };
  
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  );
};

// API Service
class APIService {
  static async get(endpoint) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) throw new Error('API request failed');
    return response.json();
  }
  
  static async post(endpoint, data) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }
    return response.json();
  }
  
  // Capital APIs
  static getCapital() {
    return this.get('/api/v1/capital/current');
  }
  
  static setCapital(amount) {
    return this.post('/api/v1/capital/set', { monthly_capital: amount });
  }
  
  static generateBasePlan() {
    return this.post('/api/v1/capital/generate-base-plan', {});
  }
  
  // Decision APIs
  static getTodayDecision() {
    return this.get('/api/v1/decision/today');
  }
  
  static getDecisionHistory(limit = 10) {
    return this.get(`/api/v1/decision/history?limit=${limit}`);
  }
  
  // Investment APIs
  static executeInvestment(type, data) {
    return this.post(`/api/v1/invest/${type}`, data);
  }
  
  static checkAllowedInvestments() {
    return this.get('/api/v1/invest/today/allowed');
  }
  
  // Portfolio APIs
  static getPortfolioSummary() {
    return this.get('/api/v1/portfolio/summary');
  }
  
  static getHoldings() {
    return this.get('/api/v1/portfolio/holdings');
  }
  
  static getPnl() {
    return this.get('/api/v1/portfolio/pnl');
  }

  static getBrokerHoldings() {
    return this.get('/api/v1/portfolio/broker-holdings');
  }

  static getInvestmentHistory(bucket = 'all', limit = 20) {
    return this.get(`/api/v1/invest/history/${bucket}?limit=${limit}`);
  }
  
  // Config APIs
  static getETFs() {
    return this.get('/api/v1/config/etfs');
  }
  
  static getRules() {
    return this.get('/api/v1/config/rules');
  }
  
  static getAllocations() {
    return Promise.all([
      this.get('/api/v1/config/allocations/base'),
      this.get('/api/v1/config/allocations/tactical')
    ]);
  }

  // Market Data APIs
  static getMarketDataStatus() {
    return this.get('/api/v1/market-data/status');
  }

  static getMarketDataTrace() {
    return this.get('/api/v1/market-data/trace');
  }

  static setUpstoxToken(token) {
    return this.post('/api/v1/market-data/upstox/token', { token, source: 'frontend' });
  }

  // Options APIs
  static getOptionsProjectCheck() {
    return this.get('/api/v1/options/project-check');
  }

  static getOptionsState() {
    return this.get('/api/v1/options/state');
  }

  static setOptionsCapital(monthlyCapital, month = null) {
    return this.post('/api/v1/options/capital', {
      monthly_capital: monthlyCapital,
      month
    });
  }

  static topupOptionsCapital(topupAmount, month = null) {
    return this.post('/api/v1/options/capital/topup', {
      topup_amount: topupAmount,
      month
    });
  }
}

// Dashboard Header Component
const DashboardHeader = ({ user = 'Investor', onMenuClick }) => {
  return (
    <header className="glass sticky top-0 z-50 border-b border-white/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <button onClick={onMenuClick} className="lg:hidden">
              <Menu className="h-6 w-6 text-ink-1" />
            </button>
            <div className="flex items-center space-x-3">
              <div className="bg-primary-600 p-2 rounded-xl shadow-sm">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-ink-0 title-font">ETF Assistant</h1>
                <p className="text-xs text-ink-1/70">Disciplined Indian ETF Investing</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="p-2 hover:bg-white/60 rounded-lg relative">
              <Bell className="h-5 w-5 text-ink-1" />
              <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="p-2 hover:bg-white/60 rounded-lg">
              <Settings className="h-5 w-5 text-ink-1" />
            </button>
            <div className="hidden sm:flex items-center space-x-2 bg-white/70 px-3 py-2 rounded-lg border border-white/60">
              <div className="h-8 w-8 bg-primary-600 rounded-full flex items-center justify-center text-white font-medium">
                {user[0]}
              </div>
              <span className="text-sm font-medium text-ink-1">{user}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

// Stats Card Component
const StatsCard = ({ title, value, change, icon: Icon, trend, subtitle }) => {
  const isPositive = trend === 'up';
  const isNeutral = trend === 'neutral';
  
  return (
    <Card className="p-6 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <h3 className="text-2xl font-bold text-gray-900 mt-2">{value}</h3>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
          {change && (
            <div className={`flex items-center mt-2 text-sm ${
              isPositive ? 'text-green-600' : isNeutral ? 'text-gray-600' : 'text-red-600'
            }`}>
              {isPositive && <ArrowUpRight className="h-4 w-4 mr-1" />}
              {!isPositive && !isNeutral && <ArrowDownRight className="h-4 w-4 mr-1" />}
              {isNeutral && <Minus className="h-4 w-4 mr-1" />}
              <span className="font-medium">{change}</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${
          isPositive ? 'bg-green-100' : isNeutral ? 'bg-gray-100' : 'bg-red-100'
        }`}>
          <Icon className={`h-6 w-6 ${
            isPositive ? 'text-green-600' : isNeutral ? 'text-gray-600' : 'text-red-600'
          }`} />
        </div>
      </div>
    </Card>
  );
};

// Today's Decision Component
const TodayDecision = ({ decision, onInvest }) => {
  if (!decision) {
    return (
      <Card className="p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Clock className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-semibold text-ink-0">Today's Decision</h2>
        </div>
        <div className="text-center py-8">
          <Activity className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p className="text-ink-1/80">No decision generated yet</p>
          <p className="text-sm text-ink-1/60 mt-1">Decision will be generated at 03:15 PM IST</p>
        </div>
      </Card>
    );
  }
  
  const decisionIcons = {
    'NONE': { icon: Minus, color: 'gray', bg: 'bg-stone-100', text: 'text-stone-600' },
    'SMALL': { icon: TrendingDown, color: 'amber', bg: 'bg-amber-100', text: 'text-amber-700' },
    'MEDIUM': { icon: TrendingDown, color: 'orange', bg: 'bg-orange-100', text: 'text-orange-700' },
    'FULL': { icon: TrendingDown, color: 'red', bg: 'bg-rose-100', text: 'text-rose-700' }
  };
  
  const decisionType = decision.decision_type || 'NONE';
  const { icon: Icon, bg, text } = decisionIcons[decisionType];
  
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Target className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Today's Decision</h2>
        </div>
        <Badge variant={decisionType === 'NONE' ? 'default' : 'warning'}>
          {decision.date}
        </Badge>
      </div>
      
      <div className="space-y-4">
        <div className={`p-4 rounded-lg ${bg} flex items-center justify-between`}>
          <div className="flex items-center space-x-3">
            <Icon className={`h-8 w-8 ${text}`} />
            <div>
            <p className="text-sm font-medium text-ink-1/70">Decision Type</p>
              <p className={`text-xl font-bold ${text}`}>{decisionType}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-ink-1/70">NIFTY Change</p>
            <p className={`text-xl font-bold ${text}`}>{decision.nifty_change_pct}%</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-sky-50 p-4 rounded-lg">
            <p className="text-sm text-ink-1/70 mb-1">Suggested Amount</p>
            <p className="text-lg font-bold text-ink-0">₹{decision.suggested_total_amount?.toLocaleString()}</p>
          </div>
          <div className="bg-emerald-50 p-4 rounded-lg">
            <p className="text-sm text-ink-1/70 mb-1">Investable</p>
            <p className="text-lg font-bold text-emerald-700">₹{decision.actual_investable_amount?.toLocaleString()}</p>
          </div>
        </div>
        
        <div className="border-t pt-4">
          <p className="text-sm font-medium text-ink-1 mb-2">Capital Remaining</p>
          <div className="flex justify-between items-center text-sm">
            <span className="text-ink-1/70">📊 Base:</span>
            <span className="font-medium">₹{decision.remaining_base_capital?.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center text-sm mt-1">
            <span className="text-ink-1/70">⚡ Tactical:</span>
            <span className="font-medium">₹{decision.remaining_tactical_capital?.toLocaleString()}</span>
          </div>
        </div>
        
        {decision.explanation && (
          <div className="bg-stone-50 p-3 rounded-lg">
            <p className="text-sm text-ink-1/80">{decision.explanation}</p>
          </div>
        )}
        
        {decisionType !== 'NONE' && (
          <Button variant="primary" className="w-full" onClick={() => onInvest('tactical')}>
            <Zap className="h-4 w-4 mr-2 inline" />
            Execute Tactical Investment
          </Button>
        )}
      </div>
    </Card>
  );
};

// Capital Overview Component
const CapitalOverview = ({ capital, onSetCapital, onViewBasePlan }) => {
  if (!capital) {
    return (
      <Card className="p-6">
        <div className="text-center py-8">
          <Wallet className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p className="text-ink-1/80 mb-4">No capital configured</p>
          <Button variant="primary" onClick={onSetCapital}>
            <Plus className="h-4 w-4 mr-2 inline" />
            Set Monthly Capital
          </Button>
        </div>
      </Card>
    );
  }
  
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Wallet className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Capital Overview</h2>
        </div>
        <Badge variant="info">{capital.month}</Badge>
      </div>
      
      <div className="space-y-4">
        <div className="bg-gradient-to-br from-primary-50 to-amber-50 p-4 rounded-lg">
          <p className="text-sm text-ink-1/70 mb-1">Total Monthly Capital</p>
          <p className="text-3xl font-bold text-ink-0">₹{capital.monthly_capital?.toLocaleString()}</p>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-sky-50 p-3 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-ink-1/70">Base (60%)</span>
              <Shield className="h-4 w-4 text-sky-600" />
            </div>
            <p className="text-lg font-bold text-ink-0">₹{capital.base_capital?.toLocaleString()}</p>
            <p className="text-xs text-ink-1/60 mt-1">Systematic investing</p>
          </div>
          
          <div className="bg-amber-50 p-3 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-ink-1/70">Tactical (40%)</span>
              <Zap className="h-4 w-4 text-amber-600" />
            </div>
            <p className="text-lg font-bold text-ink-0">₹{capital.tactical_capital?.toLocaleString()}</p>
            <p className="text-xs text-ink-1/60 mt-1">Signal-driven</p>
          </div>
        </div>
        
        <div className="border-t pt-3 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-ink-1/70">Trading Days:</span>
            <span className="font-medium">{capital.trading_days}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-ink-1/70">Daily Tranche:</span>
            <span className="font-medium">₹{capital.daily_tranche?.toLocaleString()}</span>
          </div>
          {capital.carry_forward_applied && (
            <div className="flex justify-between">
              <span className="text-ink-1/70">Carry Forward:</span>
              <span className="font-medium">
                ₹{((capital.carry_forward_base || 0) + (capital.carry_forward_tactical || 0)).toLocaleString()}
              </span>
            </div>
          )}
          {capital.total_invested !== undefined && (
            <>
              <div className="flex justify-between">
                <span className="text-ink-1/70">Allocated:</span>
                <span className="font-medium">₹{capital.monthly_capital?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-1/70">Invested:</span>
                <span className="font-medium text-emerald-700">₹{capital.total_invested?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-1/70">Remaining:</span>
                <span className="font-medium">₹{capital.total_remaining?.toLocaleString()}</span>
              </div>
            </>
          )}
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm" onClick={onViewBasePlan}>
            <BarChart3 className="h-4 w-4 mr-1 inline" />
            Base Plan
          </Button>
          <Button variant="secondary" size="sm" onClick={onSetCapital}>
            <Settings className="h-4 w-4 mr-1 inline" />
            Update
          </Button>
        </div>
      </div>
    </Card>
  );
};

// Market Data Status Component
const MarketDataStatus = ({ status, onUpdateToken }) => {
  if (!status) {
    return (
      <Card className="p-6">
        <div className="text-center py-6">
          <Activity className="h-10 w-10 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-600">Loading market data status...</p>
        </div>
      </Card>
    );
  }

  const provider = status.provider || 'unknown';
  const upstox = status.upstox;
  const needsRefresh = upstox?.needs_refresh;
  const lastUpdated = upstox?.last_updated ? new Date(upstox.last_updated).toLocaleString() : 'Never';
  const maskedToken = upstox?.masked_token || 'Not set';
  const apiKeySet = status.upstox_api_key_configured;
  const apiSecretSet = status.upstox_api_secret_configured;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Shield className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Market Data</h2>
        </div>
        <Badge variant={provider === 'upstox' ? 'info' : 'default'}>
          {provider.toUpperCase()}
        </Badge>
      </div>

      <div className="space-y-3 text-sm">
        <div className="flex justify-between">
          <span className="text-ink-1/70">Fallbacks:</span>
          <span className="font-medium">{(status.fallback_providers || []).join(", ") || 'None'}</span>
        </div>

        {provider === 'upstox' && (
          <>
            <div className="flex justify-between">
              <span className="text-ink-1/70">API Key:</span>
              <Badge variant={apiKeySet ? 'success' : 'warning'}>
                {apiKeySet ? 'Set' : 'Missing'}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-1/70">API Secret:</span>
              <Badge variant={apiSecretSet ? 'success' : 'warning'}>
                {apiSecretSet ? 'Set' : 'Missing'}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-1/70">Token:</span>
              <span className="font-mono">{maskedToken}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-1/70">Last Updated:</span>
              <span className="font-medium">{lastUpdated}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-1/70">Refresh Needed:</span>
              <Badge variant={needsRefresh ? 'danger' : 'success'}>
                {needsRefresh ? 'Yes' : 'No'}
              </Badge>
            </div>
            <Button
              variant={needsRefresh ? 'danger' : 'secondary'}
              size="sm"
              onClick={onUpdateToken}
            >
              Update Token
            </Button>
          </>
        )}
      </div>
    </Card>
  );
};

// Market Data Trace Component
const MarketDataTrace = ({ trace, onRefresh }) => {
  if (!trace) {
    return (
      <Card className="p-6">
        <div className="text-center py-6">
          <Activity className="h-10 w-10 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-600">Loading data sources...</p>
        </div>
      </Card>
    );
  }

  const prices = trace.prices || {};
  const indices = trace.indices || {};
  const priceRows = Object.entries(prices);
  const indexRows = Object.entries(indices);
  const getIndexRowClass = (value) => {
    if (value === null || value === undefined) {
      return "bg-sky-50";
    }
    const num = Number(value);
    if (Number.isNaN(num)) {
      return "bg-sky-50";
    }
    if (num > 0) return "bg-emerald-50";
    if (num < 0) return "bg-rose-50";
    return "bg-sky-50";
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Activity className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Data Sources</h2>
        </div>
        <Button variant="secondary" size="sm" onClick={onRefresh}>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      <div className="space-y-4 text-sm">
        <div>
          <p className="text-xs uppercase tracking-wide text-ink-1/60 mb-2">ETF Prices</p>
          <div className="space-y-2">
            {priceRows.map(([symbol, row]) => (
              <div key={symbol} className="flex items-center justify-between bg-stone-50 rounded-lg px-3 py-2">
                <div className="font-medium text-ink-0">{symbol}</div>
                <div className="flex items-center space-x-3">
                  <span className="text-ink-1/70">₹{row.price ?? '-'}</span>
                  <Badge variant="info">{(row.source || 'unknown').toUpperCase()}</Badge>
                </div>
              </div>
            ))}
            {priceRows.length === 0 && (
              <div className="text-ink-1/60">No prices available.</div>
            )}
          </div>
        </div>

        <div>
          <p className="text-xs uppercase tracking-wide text-ink-1/60 mb-2">Underlying Index Changes</p>
          <div className="space-y-2">
            {indexRows.map(([indexName, row]) => (
              <div
                key={indexName}
                className={`flex items-center justify-between rounded-lg px-3 py-2 ${getIndexRowClass(row.change_pct)}`}
              >
                <div className="font-medium text-ink-0">{indexName}</div>
                <div className="flex items-center space-x-3">
                  <span className="text-ink-1/70">{row.change_pct ?? '-' }%</span>
                  <Badge variant="info">{(row.source || 'unknown').toUpperCase()}</Badge>
                </div>
              </div>
            ))}
            {indexRows.length === 0 && (
              <div className="text-ink-1/60">No index data available.</div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};

const OptionsTradingOverview = ({ projectCheck, onInitCapital, onTopupCapital }) => {
  if (!projectCheck) {
    return (
      <Card className="p-6">
        <div className="text-center py-6 text-ink-1/60">Options trading status unavailable.</div>
      </Card>
    );
  }

  const risk = projectCheck.risk_limits || {};
  const riskState = risk.risk_state || {};
  const futuresCfg = projectCheck.futures_config || {};
  const initialized = Boolean(projectCheck.capital_initialized_for_month);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-ink-0">Options Trading</h2>
        <Badge variant={projectCheck.enabled ? 'success' : 'warning'}>
          {projectCheck.enabled ? 'Enabled' : 'Disabled'}
        </Badge>
      </div>

      <div className="space-y-3 text-sm">
        <div className="flex justify-between">
          <span className="text-ink-1/70">Month:</span>
          <span className="font-medium">{projectCheck.month || '-'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Trading Capital:</span>
          <span className="font-medium">₹{Number(projectCheck.monthly_capital || 0).toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Min Score:</span>
          <span className="font-medium">{projectCheck.project_min_score || '-'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Max Trades/Day:</span>
          <span className="font-medium">{risk.max_trades_per_day ?? '-'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Daily Loss Cap:</span>
          <span className="font-medium">₹{Number(risk.max_loss_per_day || 0).toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Used Risk Today:</span>
          <span className="font-medium">₹{Number(riskState.risk_used || 0).toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-ink-1/70">Futures Gate Required:</span>
          <Badge variant={futuresCfg.required ? 'warning' : 'success'}>
            {futuresCfg.required ? 'Yes' : 'No (Options-only)'}
          </Badge>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <Button
          variant="secondary"
          size="sm"
          className="w-full"
          onClick={onInitCapital}
          disabled={initialized}
        >
          {initialized ? 'Monthly Capital Locked' : 'Initialize Monthly Capital'}
        </Button>
        <Button variant="outline" size="sm" className="w-full" onClick={onTopupCapital}>
          Add Extra Capital (Top-up)
        </Button>
      </div>
    </Card>
  );
};

const OptionsSignalsCard = ({ optionsState }) => {
  if (!optionsState) {
    return (
      <Card className="p-6">
        <div className="text-center py-6 text-ink-1/60">Options signals unavailable.</div>
      </Card>
    );
  }

  const signalsMap = optionsState.signals || {};
  const allSignals = Object.values(signalsMap).flat().slice(-8).reverse();

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-ink-0">Recent Options Signals</h2>
        <Badge variant="info">{allSignals.length}</Badge>
      </div>
      <div className="space-y-2 text-sm max-h-72 overflow-y-auto pr-1">
        {allSignals.length === 0 && (
          <div className="text-ink-1/60">No signals yet.</div>
        )}
        {allSignals.map((row, idx) => (
          <div key={`${row.signal_time || row.ts || idx}-${idx}`} className="bg-stone-50 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-ink-0">{row.symbol} {row.option_side || ''}</span>
              <Badge variant={row.signal === 'BUY_CE' ? 'success' : 'warning'}>{row.signal}</Badge>
            </div>
            <div className="grid grid-cols-3 gap-2 mt-2 text-xs text-ink-1/70">
              <div>Entry: ₹{row.entry}</div>
              <div>SL: ₹{row.stop_loss}</div>
              <div>TGT: ₹{row.target}</div>
            </div>
            <div className="text-xs text-ink-1/60 mt-1">
              Score: {row.confidence_project ?? row.confidence_score ?? '-'} / 100
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// Portfolio Summary Component
const PortfolioSummary = ({ portfolio }) => {
  if (!portfolio) {
    return (
      <Card className="p-6">
        <div className="text-center py-8">
          <PieChart className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-600">No investments yet</p>
        </div>
      </Card>
    );
  }
  
  const pnlPositive = portfolio.unrealized_pnl >= 0;
  
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <PieChart className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Portfolio</h2>
        </div>
        <Button variant="secondary" size="sm">
          <ExternalLink className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-stone-50 p-4 rounded-lg">
            <p className="text-sm text-ink-1/70 mb-1">Total Invested</p>
            <p className="text-xl font-bold text-ink-0">₹{portfolio.total_invested?.toLocaleString()}</p>
          </div>
          <div className="bg-sky-50 p-4 rounded-lg">
            <p className="text-sm text-ink-1/70 mb-1">Current Value</p>
            <p className="text-xl font-bold text-sky-700">₹{portfolio.current_value?.toLocaleString()}</p>
          </div>
        </div>
        
        <div className={`p-4 rounded-lg ${pnlPositive ? 'bg-green-50' : 'bg-red-50'}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-ink-1/70 mb-1">Unrealized P&L</p>
              <p className={`text-2xl font-bold ${pnlPositive ? 'text-green-600' : 'text-red-600'}`}>
                {pnlPositive ? '+' : ''}₹{portfolio.unrealized_pnl?.toLocaleString()}
              </p>
            </div>
            <div className="text-right">
              <p className={`text-3xl font-bold ${pnlPositive ? 'text-green-600' : 'text-red-600'}`}>
                {pnlPositive ? '+' : ''}{portfolio.pnl_percentage?.toFixed(2)}%
              </p>
            </div>
          </div>
        </div>
        {portfolio.realized_pnl !== undefined && portfolio.realized_pnl !== null && (
          <div className="bg-amber-50 p-4 rounded-lg">
            <p className="text-sm text-ink-1/70 mb-1">Realized P&L</p>
            <p className="text-xl font-bold text-amber-700">
              {portfolio.realized_pnl >= 0 ? '+' : ''}₹{portfolio.realized_pnl?.toLocaleString()}
            </p>
          </div>
        )}
        {portfolio.prices_missing && portfolio.prices_missing.length > 0 && (
          <div className="text-xs text-ink-1/70">
            Prices missing for: {portfolio.prices_missing.join(", ")}
          </div>
        )}
      </div>
    </Card>
  );
};

const BrokerHoldingsCard = ({ broker, lastSyncedAt }) => {
  if (!broker) {
    return (
      <Card className="p-6">
        <div className="text-center py-6 text-ink-1/60">Broker holdings unavailable.</div>
      </Card>
    );
  }

  if (broker.status !== "ok") {
    return (
      <Card className="p-6">
        <div className="text-sm text-ink-1/70">
          Broker holdings unavailable.
        </div>
        {broker.error && (
          <div className="text-xs text-ink-1/60 mt-2">{broker.error}</div>
        )}
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-ink-0">Broker Holdings (Upstox)</h2>
        <div className="flex items-center space-x-3 text-sm text-ink-1/60">
          <div>Value: ₹{broker.total_value?.toLocaleString()}</div>
          <Badge variant="info">
            Total PnL {broker.total_pnl >= 0 ? "+" : ""}₹{broker.total_pnl?.toLocaleString()}
          </Badge>
        </div>
      </div>
      {lastSyncedAt && (
        <div className="text-xs text-ink-1/60 mb-3">Last synced at {lastSyncedAt}</div>
      )}
      <div className="space-y-2 text-sm">
        {broker.holdings.map((row) => (
          <div key={row.symbol} className="flex items-center justify-between bg-stone-50 rounded-lg px-3 py-2">
            <div className="font-medium text-ink-0">{row.symbol}</div>
            <div className="flex items-center space-x-3 text-ink-1/70">
              <span>{row.quantity} units</span>
              <span>₹{row.last_price}</span>
              <span>₹{row.current_value?.toLocaleString()}</span>
            </div>
          </div>
        ))}
        {broker.holdings.length === 0 && (
          <div className="text-ink-1/60">No broker holdings found.</div>
        )}
      </div>
    </Card>
  );
};

const CombinedPortfolioSummary = ({ appPortfolio, brokerPortfolio }) => {
  if (!appPortfolio || !brokerPortfolio || brokerPortfolio.status !== "ok") {
    return null;
  }

  const combinedValue = (appPortfolio.current_value || 0) + (brokerPortfolio.total_value || 0);
  const combinedInvested = (appPortfolio.total_invested || 0) + (brokerPortfolio.total_value || 0);
  const combinedPnl = (appPortfolio.unrealized_pnl || 0) + (brokerPortfolio.total_pnl || 0);
  const combinedPnlPct = combinedInvested > 0 ? (combinedPnl / combinedInvested) * 100 : 0;
  const pnlPositive = combinedPnl >= 0;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-ink-0">Combined Portfolio (App + Broker)</h2>
        <Badge variant="info">Read-only</Badge>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-stone-50 rounded-xl p-4">
          <p className="text-xs text-ink-1/60 uppercase tracking-wide">Estimated Invested</p>
          <p className="text-xl font-bold text-ink-0">₹{combinedInvested.toLocaleString()}</p>
        </div>
        <div className="bg-sky-50 rounded-xl p-4">
          <p className="text-xs text-ink-1/60 uppercase tracking-wide">Current Value</p>
          <p className="text-xl font-bold text-sky-700">₹{combinedValue.toLocaleString()}</p>
        </div>
        <div className="bg-emerald-50 rounded-xl p-4">
          <p className="text-xs text-ink-1/60 uppercase tracking-wide">PnL</p>
          <p className={`text-xl font-bold ${pnlPositive ? "text-emerald-700" : "text-rose-700"}`}>
            {pnlPositive ? "+" : ""}₹{combinedPnl.toLocaleString()}
          </p>
          <p className={`text-sm ${pnlPositive ? "text-emerald-700" : "text-rose-700"}`}>
            {pnlPositive ? "+" : ""}{combinedPnlPct.toFixed(2)}%
          </p>
        </div>
      </div>
    </Card>
  );
};

// Trade History Panel
const TradeHistoryPanel = ({ history, filter, onFilterChange, page, onPageChange, pageSize }) => {
  if (!history) {
    return (
      <Card className="p-6">
        <div className="text-center py-6">
          <Activity className="h-10 w-10 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-600">Loading trade history...</p>
        </div>
      </Card>
    );
  }

  if (!history.investments || history.investments.length === 0) {
    return (
      <Card className="p-6">
        <div className="text-center py-6">
          <Activity className="h-10 w-10 text-gray-300 mx-auto mb-2" />
          <p className="text-gray-600">No trades recorded yet</p>
        </div>
      </Card>
    );
  }

  const allInvestments = history.investments;
  const filteredInvestments =
    filter === 'all'
      ? allInvestments
      : allInvestments.filter((inv) => inv.bucket === filter);
  const totalBaseSpent = allInvestments
    .filter((inv) => inv.bucket === 'base')
    .reduce((sum, inv) => sum + Number(inv.total || 0), 0);
  const totalTacticalSpent = allInvestments
    .filter((inv) => inv.bucket === 'tactical')
    .reduce((sum, inv) => sum + Number(inv.total || 0), 0);

  const totalPages = Math.max(1, Math.ceil(filteredInvestments.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const startIndex = (currentPage - 1) * pageSize;
  const pagedInvestments = filteredInvestments.slice(startIndex, startIndex + pageSize);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Activity className="h-5 w-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-ink-0">Recent Trades</h2>
        </div>
        <Badge variant="info">{history.count}</Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
        <div className="bg-stone-50 p-3 rounded-lg">
          <p className="text-ink-1/60">Base Spent</p>
          <p className="font-semibold">₹{totalBaseSpent.toLocaleString()}</p>
        </div>
        <div className="bg-amber-50 p-3 rounded-lg">
          <p className="text-ink-1/60">Tactical Spent</p>
          <p className="font-semibold">₹{totalTacticalSpent.toLocaleString()}</p>
        </div>
      </div>

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Button
            variant={filter === 'all' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onFilterChange('all')}
          >
            All
          </Button>
          <Button
            variant={filter === 'base' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onFilterChange('base')}
          >
            Base
          </Button>
          <Button
            variant={filter === 'tactical' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => onFilterChange('tactical')}
          >
            Tactical
          </Button>
        </div>
        <div className="text-xs text-ink-1/60">
          Page {currentPage} of {totalPages}
        </div>
      </div>

      <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
        {pagedInvestments.map((inv) => (
          <div key={inv.id} className="border border-gray-200 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="font-bold text-ink-0">{inv.etf_symbol}</span>
                <Badge variant={inv.bucket === 'tactical' ? 'warning' : 'default'}>
                  {inv.bucket}
                </Badge>
              </div>
              <span className="text-xs text-ink-1/60">{inv.date}</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-sm mt-2">
              <div>
                <p className="text-ink-1/60">Units</p>
                <p className="font-medium">{inv.units}</p>
              </div>
              <div>
                <p className="text-ink-1/60">Price</p>
                <p className="font-medium">₹{Number(inv.price).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-ink-1/60">Total</p>
                <p className="font-medium">₹{Number(inv.total).toLocaleString()}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between mt-4">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage <= 1}
        >
          Prev
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage >= totalPages}
        >
          Next
        </Button>
      </div>
    </Card>
  );
};

// Set Capital Modal
const SetCapitalModal = ({ isOpen, onClose, onSubmit }) => {
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(false);
  
  if (!isOpen) return null;
  
  const handleSubmit = async () => {
    if (!amount || parseFloat(amount) < 1000) {
      alert('Minimum capital is ₹1,000');
      return;
    }
    
    setLoading(true);
    try {
      await onSubmit(parseFloat(amount));
      setAmount('');
      onClose();
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <Card className="max-w-md w-full p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">Set Monthly Capital</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Monthly Investment Amount
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">₹</span>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="50000"
                className="w-full pl-8 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <p className="text-sm text-gray-500 mt-2">Minimum: ₹1,000</p>
          </div>
          
          {amount && parseFloat(amount) >= 1000 && (
            <div className="bg-blue-50 p-4 rounded-lg space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Base (60%):</span>
                <span className="font-medium">₹{(parseFloat(amount) * 0.6).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Tactical (40%):</span>
                <span className="font-medium">₹{(parseFloat(amount) * 0.4).toLocaleString()}</span>
              </div>
            </div>
          )}
          
          <div className="flex space-x-3">
            <Button variant="secondary" className="flex-1" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" className="flex-1" onClick={handleSubmit} disabled={loading}>
              {loading ? 'Setting...' : 'Confirm'}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

// Set Upstox Token Modal
const SetTokenModal = ({ isOpen, onClose, onSubmit }) => {
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async () => {
    if (!token || token.length < 10) {
      alert('Please enter a valid Upstox token');
      return;
    }

    setLoading(true);
    try {
      await onSubmit(token.trim());
      setToken('');
      onClose();
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <Card className="max-w-md w-full p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">Update Upstox Token</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upstox Access Token
            </label>
            <textarea
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Paste your Upstox access token here"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent h-28"
            />
            <p className="text-sm text-gray-500 mt-2">
              Tokens expire daily. Update before market opens (9:00 AM).
            </p>
          </div>

          <div className="flex space-x-3">
            <Button variant="secondary" className="flex-1" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" className="flex-1" onClick={handleSubmit} disabled={loading}>
              {loading ? 'Saving...' : 'Save Token'}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

// Base Plan Modal
const BasePlanModal = ({ isOpen, onClose, plan }) => {
  if (!isOpen) return null;
  
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto"
      onClick={onClose}
    >
      <Card className="max-w-3xl w-full p-6 my-8" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">Base Investment Plan</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        
        {plan ? (
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-primary-50 to-amber-50 p-4 rounded-lg">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Base Capital</p>
                  <p className="text-lg font-bold">₹{plan.base_capital?.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Total Investable</p>
                  <p className="text-lg font-bold text-green-600">₹{plan.total_actual?.toLocaleString()}</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-900">ETF-wise Breakdown</h4>
              {Object.entries(plan.base_plan || {}).map(([symbol, details]) => (
                <div key={symbol} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <h5 className="font-bold text-gray-900">{symbol}</h5>
                      <Badge variant="default">{details.allocation_pct ?? 0}%</Badge>
                    </div>
                    {(!details.ltp || details.status === 'price_unavailable') && (
                      <Badge variant="danger">Price N/A</Badge>
                    )}
                  </div>
                  
                  {!!details.ltp && details.status !== 'price_unavailable' && (
                    <div className="grid grid-cols-3 gap-3 text-sm mt-3">
                      <div>
                        <p className="text-gray-600">LTP</p>
                        <p className="font-medium">₹{details.ltp?.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Units</p>
                        <p className="font-medium text-primary-600">{details.recommended_units}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Amount</p>
                        <p className="font-medium">₹{details.actual_amount?.toLocaleString()}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-sm text-gray-700">
                <Info className="h-4 w-4 inline mr-1" />
                Execute these investments gradually throughout the month. Can invest on any trading day.
              </p>
            </div>

            <div className="flex justify-end">
              <Button variant="secondary" onClick={onClose}>
                Close
              </Button>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <RefreshCw className="h-12 w-12 text-gray-300 mx-auto mb-3 animate-spin" />
            <p className="text-gray-600">Loading base plan...</p>
          </div>
        )}
      </Card>
    </div>
  );
};

// Quick Trade Modal
const QuickTradeModal = ({ isOpen, onClose, tradeType, symbols, onSubmit, basePlan, decision, history }) => {
  const [etfSymbol, setEtfSymbol] = useState('');
  const [units, setUnits] = useState('');
  const [price, setPrice] = useState('');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const tacticalBlocked = tradeType === 'tactical' && (!decision || decision.decision_type === 'NONE');
  const todayStr = new Date().toISOString().slice(0, 10);
  const currentMonth = todayStr.slice(0, 7);
  const alreadyExecuted = (() => {
    if (!history?.investments || !etfSymbol) return false;
    const symbol = etfSymbol.toUpperCase();
    if (tradeType === 'base') {
      return history.investments.some(
        (inv) =>
          inv.bucket === 'base' &&
          inv.etf_symbol === symbol &&
          inv.date &&
          inv.date.startsWith(currentMonth)
      );
    }
    if (tradeType === 'tactical') {
      return history.investments.some(
        (inv) =>
          inv.bucket === 'tactical' &&
          inv.etf_symbol === symbol &&
          inv.date === todayStr
      );
    }
    return false;
  })();
  const recommendedUnits = basePlan?.base_plan?.[etfSymbol?.toUpperCase()]?.recommended_units || 0;

  useEffect(() => {
    if (!isOpen) {
      setEtfSymbol('');
      setUnits('');
      setPrice('');
      setNotes('');
      setLoading(false);
    }
  }, [isOpen]);

  useEffect(() => {
    if (tradeType !== 'base') return;
    if (!etfSymbol || !basePlan?.base_plan) return;
    const planEntry = basePlan.base_plan[etfSymbol.toUpperCase()];
    if (planEntry && planEntry.recommended_units > 0) {
      setUnits(String(planEntry.recommended_units));
    }
  }, [tradeType, etfSymbol, basePlan]);

  if (!isOpen) return null;

  const handleSubmit = async () => {
    if (!etfSymbol || !units || !price) return;
    if (Number(units) < 1) {
      alert('Units must be at least 1');
      return;
    }
    if (alreadyExecuted) {
      const message =
        tradeType === 'base'
          ? `Base trade already executed for ${etfSymbol.toUpperCase()} this month.`
          : `Tactical trade already executed for ${etfSymbol.toUpperCase()} today.`;
      console.warn(message);
      alert(message);
      return;
    }
    setLoading(true);
    try {
      await onSubmit({
        etf_symbol: etfSymbol.toUpperCase(),
        units: Number(units),
        executed_price: Number(price),
        notes: notes || undefined
      });
      onClose();
    } catch (err) {
      alert(err.message || 'Failed to execute trade');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <Card className="max-w-md w-full p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">
            {tradeType === 'base' ? 'Execute Base Trade' : 'Execute Tactical Trade'}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-4">
          {tacticalBlocked && (
            <div className="bg-amber-50 text-amber-800 text-sm p-3 rounded-lg">
              Tactical trades are blocked because today's decision is NONE.
            </div>
          )}
          {alreadyExecuted && (
            <div className="bg-rose-50 text-rose-800 text-sm p-3 rounded-lg">
              {tradeType === 'base'
                ? 'Base trade already executed for this ETF this month.'
                : 'Tactical trade already executed for this ETF today.'}
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">ETF Symbol</label>
            <input
              list="etf-symbols"
              value={etfSymbol}
              onChange={(e) => setEtfSymbol(e.target.value)}
              placeholder="NIFTYBEES"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
            <datalist id="etf-symbols">
              {symbols.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
            {tradeType === 'base' && etfSymbol && (
              <p className="text-xs text-gray-500 mt-1">
                Recommended units: <span className="font-medium text-gray-700">{recommendedUnits || 0}</span>
              </p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Units</label>
              <input
                type="number"
                min="1"
                value={units}
                onChange={(e) => setUnits(e.target.value)}
                placeholder="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Executed Price</label>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="100.50"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Notes (optional)</label>
            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Order ID, broker note, etc."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
        </div>

        <div className="flex space-x-3 mt-6">
          <Button variant="secondary" className="flex-1" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            className="flex-1"
            onClick={handleSubmit}
            disabled={loading || !etfSymbol || !units || !price || Number(units) < 1 || tacticalBlocked || alreadyExecuted}
          >
            {loading ? 'Saving...' : 'Execute'}
          </Button>
        </div>
      </Card>
    </div>
  );
};

// Main Dashboard Component
const ETFDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [capital, setCapital] = useState(null);
  const [decision, setDecision] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [brokerPortfolio, setBrokerPortfolio] = useState(null);
  const [brokerSyncedAt, setBrokerSyncedAt] = useState(null);
  const [basePlan, setBasePlan] = useState(null);
  const [marketDataStatus, setMarketDataStatus] = useState(null);
  const [marketDataTrace, setMarketDataTrace] = useState(null);
  const [optionsProjectCheck, setOptionsProjectCheck] = useState(null);
  const [optionsState, setOptionsState] = useState(null);
  const [showCapitalModal, setShowCapitalModal] = useState(false);
  const [showBasePlanModal, setShowBasePlanModal] = useState(false);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [tradeModalOpen, setTradeModalOpen] = useState(false);
  const [tradeType, setTradeType] = useState('base');
  const [tradeHistory, setTradeHistory] = useState(null);
  const [tradeFilter, setTradeFilter] = useState('all');
  const [tradePage, setTradePage] = useState(1);
  const tradePageSize = 10;
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  useEffect(() => {
    loadDashboardData();
  }, []);
  
  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [capitalData, portfolioData, brokerData, historyData, marketStatus, marketTrace, projectCheckData, optionsStateData] = await Promise.all([
        APIService.getCapital().catch(() => null),
        APIService.getPnl().catch(() => null),
        APIService.getBrokerHoldings().catch(() => null),
        APIService.getInvestmentHistory('all', 200).catch(() => null),
        APIService.getMarketDataStatus().catch(() => null),
        APIService.getMarketDataTrace().catch(() => null),
        APIService.getOptionsProjectCheck().catch(() => null),
        APIService.getOptionsState().catch(() => null)
      ]);
      
      setCapital(capitalData);
      setPortfolio(portfolioData);
      setBrokerPortfolio(brokerData);
      if (brokerData) {
        setBrokerSyncedAt(new Date().toLocaleString());
      }
      setTradeHistory(historyData);
      setMarketDataStatus(marketStatus);
      setMarketDataTrace(marketTrace);
      setOptionsProjectCheck(projectCheckData);
      setOptionsState(optionsStateData);
      
      // Load decision
      try {
        const decisionData = await APIService.getTodayDecision();
        setDecision(decisionData);
      } catch (e) {
        setDecision(null);
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInitOptionsCapital = async () => {
    const initialized = Boolean(optionsProjectCheck?.capital_initialized_for_month);
    if (initialized) {
      alert('Monthly capital is already initialized for this month. Use top-up to add extra capital.');
      return;
    }
    const amountText = window.prompt('Initialize options monthly capital (INR):', '10000');
    if (!amountText) return;
    const amount = Number(amountText);
    if (!Number.isFinite(amount) || amount <= 0) {
      alert('Enter a valid amount.');
      return;
    }
    try {
      await APIService.setOptionsCapital(amount);
      const projectCheckData = await APIService.getOptionsProjectCheck().catch(() => null);
      setOptionsProjectCheck(projectCheckData);
    } catch (error) {
      alert(error.message || 'Failed to initialize options capital.');
    }
  };

  const handleTopupOptionsCapital = async () => {
    const amountText = window.prompt('Add extra options capital (INR):', '1000');
    if (!amountText) return;
    const amount = Number(amountText);
    if (!Number.isFinite(amount) || amount <= 0) {
      alert('Enter a valid amount.');
      return;
    }
    try {
      await APIService.topupOptionsCapital(amount);
      const projectCheckData = await APIService.getOptionsProjectCheck().catch(() => null);
      setOptionsProjectCheck(projectCheckData);
    } catch (error) {
      alert(error.message || 'Failed to top-up options capital.');
    }
  };
  
  const handleSetCapital = async (amount) => {
    await APIService.setCapital(amount);
    await loadDashboardData();
  };

  const handleSetUpstoxToken = async (token) => {
    await APIService.setUpstoxToken(token);
    const status = await APIService.getMarketDataStatus().catch(() => null);
    setMarketDataStatus(status);
    const trace = await APIService.getMarketDataTrace().catch(() => null);
    setMarketDataTrace(trace);
  };

  const refreshMarketTrace = async () => {
    const trace = await APIService.getMarketDataTrace().catch(() => null);
    setMarketDataTrace(trace);
  };
  
  const handleViewBasePlan = async () => {
    setShowBasePlanModal(true);
    try {
      const plan = await APIService.generateBasePlan();
      const entries = Object.values(plan.base_plan || {});
      const totals = entries.reduce(
        (acc, item) => {
          acc.total_allocated += item.allocated_amount || 0;
          acc.total_actual += item.actual_amount || 0;
          acc.total_unused += item.unused || 0;
          return acc;
        },
        { total_allocated: 0, total_actual: 0, total_unused: 0 }
      );
      setBasePlan({ ...plan, ...totals });
    } catch (error) {
      alert(error.message);
      setShowBasePlanModal(false);
    }
  };

  const closeBasePlan = () => {
    setShowBasePlanModal(false);
    setBasePlan(null);
  };

  const openTradeModal = async (type) => {
    setTradeType(type);
    if (type === 'base' && !basePlan) {
      try {
        const plan = await APIService.generateBasePlan();
        const entries = Object.values(plan.base_plan || {});
        const totals = entries.reduce(
          (acc, item) => {
            acc.total_allocated += item.allocated_amount || 0;
            acc.total_actual += item.actual_amount || 0;
            acc.total_unused += item.unused || 0;
            return acc;
          },
          { total_allocated: 0, total_actual: 0, total_unused: 0 }
        );
        setBasePlan({ ...plan, ...totals });
      } catch (error) {
        alert(error.message);
        return;
      }
    }
    setTradeModalOpen(true);
  };

  const handleTradeSubmit = async (data) => {
    await APIService.executeInvestment(tradeType, data);
    await loadDashboardData();
  };
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-12 w-12 text-primary-600 mx-auto mb-4 animate-spin" />
          <p className="text-ink-1/80">Loading dashboard...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen">
      <DashboardHeader onMenuClick={() => setMobileMenuOpen(!mobileMenuOpen)} />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 fade-in">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatsCard
            title="Total Invested"
            value={portfolio ? `₹${portfolio.total_invested?.toLocaleString()}` : '₹0'}
            icon={DollarSign}
            trend="neutral"
            subtitle="All-time investment"
          />
          <StatsCard
            title="Current Value"
            value={portfolio ? `₹${portfolio.current_value?.toLocaleString()}` : '₹0'}
            icon={TrendingUp}
            trend={portfolio && portfolio.pnl_percentage >= 0 ? 'up' : 'down'}
            change={portfolio ? `${portfolio.pnl_percentage?.toFixed(2)}%` : '0%'}
          />
          <StatsCard
            title="Monthly Capital"
            value={capital ? `₹${capital.monthly_capital?.toLocaleString()}` : 'Not Set'}
            icon={Wallet}
            trend="neutral"
            subtitle={capital ? capital.month : 'Configure now'}
          />
          <StatsCard
            title="Trading Days"
            value={capital ? capital.trading_days : '0'}
            icon={Calendar}
            trend="neutral"
            subtitle={capital ? `₹${capital.daily_tranche?.toLocaleString()} daily` : 'N/A'}
          />
        </div>
        
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Left Column - Decision & Activity */}
          <div className="lg:col-span-2 space-y-6">
            <TodayDecision decision={decision} onInvest={() => openTradeModal('tactical')} />
            
            {/* Quick Actions */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-ink-0 mb-4">Quick Actions</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4" onClick={() => openTradeModal('base')}>
                  <Shield className="h-5 w-5 mb-2" />
                  <span className="text-xs">Base Invest</span>
                </Button>
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4" onClick={() => openTradeModal('tactical')}>
                  <Zap className="h-5 w-5 mb-2" />
                  <span className="text-xs">Tactical</span>
                </Button>
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4" onClick={handleViewBasePlan}>
                  <BarChart3 className="h-5 w-5 mb-2" />
                  <span className="text-xs">Base Plan</span>
                </Button>
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4" onClick={loadDashboardData}>
                  <RefreshCw className="h-5 w-5 mb-2" />
                  <span className="text-xs">Refresh</span>
                </Button>
              </div>
            </Card>

            <TradeHistoryPanel
              history={tradeHistory}
              filter={tradeFilter}
              onFilterChange={(f) => {
                setTradeFilter(f);
                setTradePage(1);
              }}
              page={tradePage}
              onPageChange={setTradePage}
              pageSize={tradePageSize}
            />
            <CombinedPortfolioSummary appPortfolio={portfolio} brokerPortfolio={brokerPortfolio} />
            <PortfolioSummary portfolio={portfolio} />
            <BrokerHoldingsCard broker={brokerPortfolio} lastSyncedAt={brokerSyncedAt} />
          </div>
          
          {/* Right Column - Capital & Portfolio */}
          <div className="space-y-6">
            <CapitalOverview 
              capital={capital} 
              onSetCapital={() => setShowCapitalModal(true)}
              onViewBasePlan={handleViewBasePlan}
            />
            <MarketDataStatus
              status={marketDataStatus}
              onUpdateToken={() => setShowTokenModal(true)}
            />
            <MarketDataTrace
              trace={marketDataTrace}
              onRefresh={refreshMarketTrace}
            />
            <OptionsTradingOverview
              projectCheck={optionsProjectCheck}
              onInitCapital={handleInitOptionsCapital}
              onTopupCapital={handleTopupOptionsCapital}
            />
            <OptionsSignalsCard optionsState={optionsState} />
          </div>
        </div>
      </div>
      
      {/* Modals */}
      <SetCapitalModal 
        isOpen={showCapitalModal}
        onClose={() => setShowCapitalModal(false)}
        onSubmit={handleSetCapital}
      />

      <SetTokenModal
        isOpen={showTokenModal}
        onClose={() => setShowTokenModal(false)}
        onSubmit={handleSetUpstoxToken}
      />
      
      <BasePlanModal
        isOpen={showBasePlanModal}
        onClose={closeBasePlan}
        plan={basePlan}
      />

      <QuickTradeModal
        isOpen={tradeModalOpen}
        onClose={() => setTradeModalOpen(false)}
        tradeType={tradeType}
        basePlan={basePlan}
        decision={decision}
        history={tradeHistory}
        symbols={[
          ...new Set([
            ...(basePlan ? Object.keys(basePlan.base_plan || {}) : []),
            ...(decision?.etf_decisions || []).map((d) => d.etf_symbol),
          ]),
        ]}
        onSubmit={handleTradeSubmit}
      />
    </div>
  );
};

export default ETFDashboard;
