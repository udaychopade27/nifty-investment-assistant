import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Wallet, PieChart, Calendar, 
  DollarSign, Activity, ChevronRight, AlertCircle, CheckCircle,
  ArrowUpRight, ArrowDownRight, Minus, Plus, BarChart3, 
  RefreshCw, Bell, Settings, LogOut, Menu, X, Download,
  Target, Shield, Zap, Clock, Info, ExternalLink
} from 'lucide-react';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Utility Components
const Card = ({ children, className = '' }) => (
  <div className={`bg-white rounded-xl shadow-sm border border-gray-100 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, variant = 'primary', size = 'md', onClick, disabled, className = '' }) => {
  const variants = {
    primary: 'bg-indigo-600 hover:bg-indigo-700 text-white',
    secondary: 'bg-gray-100 hover:bg-gray-200 text-gray-900',
    success: 'bg-green-600 hover:bg-green-700 text-white',
    danger: 'bg-red-600 hover:bg-red-700 text-white',
    outline: 'border-2 border-indigo-600 text-indigo-600 hover:bg-indigo-50'
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
    default: 'bg-gray-100 text-gray-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    danger: 'bg-red-100 text-red-800',
    info: 'bg-blue-100 text-blue-800'
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
  
  static getPerformance() {
    return this.get('/api/v1/portfolio/performance');
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
}

// Dashboard Header Component
const DashboardHeader = ({ user = 'Investor', onMenuClick }) => {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <button onClick={onMenuClick} className="lg:hidden">
              <Menu className="h-6 w-6 text-gray-600" />
            </button>
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-600 p-2 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">ETF Assistant</h1>
                <p className="text-xs text-gray-500">Disciplined Indian ETF Investing</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="p-2 hover:bg-gray-100 rounded-lg relative">
              <Bell className="h-5 w-5 text-gray-600" />
              <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="p-2 hover:bg-gray-100 rounded-lg">
              <Settings className="h-5 w-5 text-gray-600" />
            </button>
            <div className="hidden sm:flex items-center space-x-2 bg-gray-100 px-3 py-2 rounded-lg">
              <div className="h-8 w-8 bg-indigo-600 rounded-full flex items-center justify-center text-white font-medium">
                {user[0]}
              </div>
              <span className="text-sm font-medium text-gray-700">{user}</span>
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
          <h2 className="text-lg font-semibold text-gray-900">Today's Decision</h2>
        </div>
        <div className="text-center py-8">
          <Activity className="h-12 w-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-600">No decision generated yet</p>
          <p className="text-sm text-gray-500 mt-1">Decision will be generated at 10:00 AM</p>
        </div>
      </Card>
    );
  }
  
  const decisionIcons = {
    'NONE': { icon: Minus, color: 'gray', bg: 'bg-gray-100', text: 'text-gray-600' },
    'SMALL': { icon: TrendingDown, color: 'yellow', bg: 'bg-yellow-100', text: 'text-yellow-600' },
    'MEDIUM': { icon: TrendingDown, color: 'orange', bg: 'bg-orange-100', text: 'text-orange-600' },
    'FULL': { icon: TrendingDown, color: 'red', bg: 'bg-red-100', text: 'text-red-600' }
  };
  
  const decisionType = decision.decision_type || 'NONE';
  const { icon: Icon, bg, text } = decisionIcons[decisionType];
  
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Target className="h-5 w-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-gray-900">Today's Decision</h2>
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
              <p className="text-sm font-medium text-gray-600">Decision Type</p>
              <p className={`text-xl font-bold ${text}`}>{decisionType}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm font-medium text-gray-600">NIFTY Change</p>
            <p className={`text-xl font-bold ${text}`}>{decision.nifty_change_pct}%</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Suggested Amount</p>
            <p className="text-lg font-bold text-gray-900">₹{decision.suggested_total_amount?.toLocaleString()}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Investable</p>
            <p className="text-lg font-bold text-green-600">₹{decision.actual_investable_amount?.toLocaleString()}</p>
          </div>
        </div>
        
        <div className="border-t pt-4">
          <p className="text-sm font-medium text-gray-700 mb-2">Capital Remaining</p>
          <div className="flex justify-between items-center text-sm">
            <span className="text-gray-600">📊 Base:</span>
            <span className="font-medium">₹{decision.remaining_base_capital?.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center text-sm mt-1">
            <span className="text-gray-600">⚡ Tactical:</span>
            <span className="font-medium">₹{decision.remaining_tactical_capital?.toLocaleString()}</span>
          </div>
        </div>
        
        {decision.explanation && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <p className="text-sm text-gray-700">{decision.explanation}</p>
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
          <p className="text-gray-600 mb-4">No capital configured</p>
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
          <Wallet className="h-5 w-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-gray-900">Capital Overview</h2>
        </div>
        <Badge variant="info">{capital.month}</Badge>
      </div>
      
      <div className="space-y-4">
        <div className="bg-gradient-to-br from-indigo-50 to-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Total Monthly Capital</p>
          <p className="text-3xl font-bold text-gray-900">₹{capital.monthly_capital?.toLocaleString()}</p>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-600">Base (60%)</span>
              <Shield className="h-4 w-4 text-blue-600" />
            </div>
            <p className="text-lg font-bold text-gray-900">₹{capital.base_capital?.toLocaleString()}</p>
            <p className="text-xs text-gray-500 mt-1">Systematic investing</p>
          </div>
          
          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-600">Tactical (40%)</span>
              <Zap className="h-4 w-4 text-purple-600" />
            </div>
            <p className="text-lg font-bold text-gray-900">₹{capital.tactical_capital?.toLocaleString()}</p>
            <p className="text-xs text-gray-500 mt-1">Signal-driven</p>
          </div>
        </div>
        
        <div className="border-t pt-3 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Trading Days:</span>
            <span className="font-medium">{capital.trading_days}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Daily Tranche:</span>
            <span className="font-medium">₹{capital.daily_tranche?.toLocaleString()}</span>
          </div>
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
          <PieChart className="h-5 w-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-gray-900">Portfolio</h2>
        </div>
        <Button variant="secondary" size="sm">
          <ExternalLink className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Total Invested</p>
            <p className="text-xl font-bold text-gray-900">₹{portfolio.total_invested?.toLocaleString()}</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Current Value</p>
            <p className="text-xl font-bold text-blue-600">₹{portfolio.current_value?.toLocaleString()}</p>
          </div>
        </div>
        
        <div className={`p-4 rounded-lg ${pnlPositive ? 'bg-green-50' : 'bg-red-50'}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Unrealized P&L</p>
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
                className="w-full pl-8 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
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

// Base Plan Modal
const BasePlanModal = ({ isOpen, onClose, plan }) => {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <Card className="max-w-3xl w-full p-6 my-8">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">Base Investment Plan</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        
        {plan ? (
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-indigo-50 to-blue-50 p-4 rounded-lg">
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
                      <Badge variant="default">{details.allocation_pct}%</Badge>
                    </div>
                    {details.status === 'price_unavailable' && (
                      <Badge variant="danger">Price N/A</Badge>
                    )}
                  </div>
                  
                  {details.status !== 'price_unavailable' && (
                    <div className="grid grid-cols-3 gap-3 text-sm mt-3">
                      <div>
                        <p className="text-gray-600">LTP</p>
                        <p className="font-medium">₹{details.ltp?.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Units</p>
                        <p className="font-medium text-indigo-600">{details.recommended_units}</p>
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

// Main Dashboard Component
const ETFDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [capital, setCapital] = useState(null);
  const [decision, setDecision] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [basePlan, setBasePlan] = useState(null);
  const [showCapitalModal, setShowCapitalModal] = useState(false);
  const [showBasePlanModal, setShowBasePlanModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  useEffect(() => {
    loadDashboardData();
  }, []);
  
  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [capitalData, portfolioData] = await Promise.all([
        APIService.getCapital().catch(() => null),
        APIService.getPortfolioSummary().catch(() => null)
      ]);
      
      setCapital(capitalData);
      setPortfolio(portfolioData);
      
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
  
  const handleSetCapital = async (amount) => {
    await APIService.setCapital(amount);
    await loadDashboardData();
  };
  
  const handleViewBasePlan = async () => {
    setShowBasePlanModal(true);
    try {
      const plan = await APIService.generateBasePlan();
      setBasePlan(plan);
    } catch (error) {
      alert(error.message);
      setShowBasePlanModal(false);
    }
  };
  
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-12 w-12 text-indigo-600 mx-auto mb-4 animate-spin" />
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <DashboardHeader onMenuClick={() => setMobileMenuOpen(!mobileMenuOpen)} />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Decision & Capital */}
          <div className="lg:col-span-2 space-y-6">
            <TodayDecision decision={decision} onInvest={() => alert('Invest feature')} />
            
            {/* Quick Actions */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4">
                  <Shield className="h-5 w-5 mb-2" />
                  <span className="text-xs">Base Invest</span>
                </Button>
                <Button variant="outline" size="sm" className="flex flex-col items-center py-4">
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
          </div>
          
          {/* Right Column - Capital & Portfolio */}
          <div className="space-y-6">
            <CapitalOverview 
              capital={capital} 
              onSetCapital={() => setShowCapitalModal(true)}
              onViewBasePlan={handleViewBasePlan}
            />
            <PortfolioSummary portfolio={portfolio} />
          </div>
        </div>
      </div>
      
      {/* Modals */}
      <SetCapitalModal 
        isOpen={showCapitalModal}
        onClose={() => setShowCapitalModal(false)}
        onSubmit={handleSetCapital}
      />
      
      <BasePlanModal
        isOpen={showBasePlanModal}
        onClose={() => setShowBasePlanModal(false)}
        plan={basePlan}
      />
    </div>
  );
};

export default ETFDashboard;
