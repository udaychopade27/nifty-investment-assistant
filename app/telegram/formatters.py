def format_execution_error(error: dict) -> str:
    code = error.get("error_code")

    if code == "DAILY_LIMIT_EXCEEDED":
        d = error["details"]
        return (
            "❌ *Investment limit reached today*\n\n"
            f"Suggested today: ₹{d['suggested_amount']}\n"
            f"Already invested: ₹{d['already_executed']}\n"
            f"Remaining: ₹{d['remaining_allowed']}"
        )

    if code == "DUPLICATE_EXECUTION":
        return "⚠️ This execution was already recorded."

    if code == "MONTHLY_TACTICAL_EXCEEDED":
        return "❌ Monthly tactical capital exhausted."

    return f"❌ {error.get('message', 'Execution failed')}"
