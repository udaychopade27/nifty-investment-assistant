"""
STRATEGY CONSTITUTION — MODULE S0

This document defines the non-negotiable constitutional principles of the
ETF Investing Assistant for the Indian stock market.

This file is intentionally immutable by design.
It contains no executable logic, no imports, no functions, and no classes.

Any change to this constitution represents a strategy governance event
and must result in a new strategy version.
"""

STRATEGY_NAME = "Indian ETF Long-Term Investing Assistant"

STRATEGY_VERSION = "S0"

STRATEGY_GOVERNANCE_MODEL = "Explicit versioned constitution with locked historical behavior"

IS_TRADING_BOT = False

AUTO_EXECUTION_ALLOWED = False

DETERMINISTIC_ONLY = True

ALLOWED_LOGIC_TYPES = (
    "rule_based",
    "deterministic",
    "fully_explainable",
)

DISALLOWED_LOGIC_TYPES = (
    "machine_learning",
    "statistical_prediction",
    "technical_indicators",
    "signals",
    "probabilistic_models",
)

PRIMARY_OBJECTIVE = "Capital preservation first, disciplined long-term wealth creation second"

RISK_PHILOSOPHY = (
    "Avoid permanent capital loss, "
    "embrace controlled volatility, "
    "prefer missed opportunity over forced exposure"
)

SOURCE_OF_TRUTH = "PostgreSQL"

IN_MEMORY_DECISIONS_ALLOWED = False

ALL_DECISIONS_MUST_BE_PERSISTED = True

ALL_DECISIONS_MUST_BE_EXPLAINABLE = True

HUMAN_IN_THE_LOOP_REQUIRED = True

HUMAN_APPROVAL_REQUIRED_FOR = (
    "capital_deployment",
    "rebalancing_execution",
    "dip_buy_execution",
)

MONTHLY_PLANNING_REQUIRED = True

CAPITAL_DEPLOYMENT_FREQUENCY = "monthly"

DIP_INVESTING_ALLOWED = True

DIP_INVESTING_TYPE = "rule_based_only"

STRATEGY_TIME_HORIZON = "long_term"

SUPPORTED_INSTRUMENTS = (
    "equity_etfs",
    "index_etfs",
)

MARKET_SCOPE = "India"

UI_LAYERS_ALLOWED = (
    "api",
    "telegram",
)

UI_LAYERS_HAVE_BUSINESS_LOGIC = False

DOMAIN_LAYER_RULES = (
    "no_database_access",
    "no_api_access",
    "pure_business_rules_only",
)

SERVICE_LAYER_ROLE = (
    "orchestrate_domain_and_persistence",
    "enforce_idempotency",
    "persist_all_decisions",
)

AUDITABILITY_REQUIREMENTS = (
    "every_decision_logged",
    "every_decision_versioned",
    "every_decision_reproducible",
)

IDEMPOTENCY_REQUIRED = True

BACKTESTING_MODE = False

REAL_TIME_TRADING = False

FAILURE_MODE = "fail_safe_no_action"

STRATEGY_MUTABILITY = "immutable_per_version"

CONSTITUTION_AUTHORITY = "strategy_owner_only"

LAST_AMENDED = "initial_creation"

LEGAL_DISCLAIMER = (
    "This system is an investment assistant, "
    "not a financial advisor, broker, or execution engine."
)
