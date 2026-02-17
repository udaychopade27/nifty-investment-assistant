"""Resolve option instrument keys from Upstox option chain."""
from __future__ import annotations

from datetime import datetime, date
from typing import Dict, Any, List, Optional

import httpx

from app.domain.services.api_token_service import ApiTokenService
from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.upstox_provider import UpstoxProvider


class OptionsChainResolver:
    def __init__(self, config_engine: ConfigEngine):
        self._config_engine = config_engine

    async def resolve(self) -> List[Dict[str, Any]]:
        options_cfg = self._config_engine.get_options_setting("options")
        md_cfg = options_cfg.get("market_data", {}) or {}
        instruments = md_cfg.get("option_instruments", []) or []
        if not instruments:
            return []
        instruments = self._prepare_instruments(options_cfg, instruments)

        market_cfg = self._config_engine.get_app_setting("market_data")
        upstox_cfg = market_cfg.get("upstox", {})
        api_base = upstox_cfg.get("api_base_url", "https://api.upstox.com")
        instrument_map: Dict[str, str] = upstox_cfg.get("instrument_keys", {}) or {}

        token = await ApiTokenService("upstox").get_token()
        if not token:
            return instruments

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        cached_ctx: Dict[str, Dict[str, Any]] = {}
        for inst in instruments:
            if not isinstance(inst, dict):
                continue
            if inst.get("key"):
                continue
            underlying = inst.get("underlying")
            step = int(inst.get("step") or 0)
            side = inst.get("side")
            strike = inst.get("strike")
            if not underlying or not side:
                continue

            underlying_key = self._resolve_underlying_key(underlying, instrument_map)
            if not underlying_key:
                continue

            ctx = cached_ctx.get(underlying_key)
            if ctx is None:
                expiry = await self._get_nearest_expiry(api_base, headers, underlying_key)
                if not expiry:
                    continue
                chain = await self._get_option_chain(api_base, headers, underlying_key, expiry)
                if not chain:
                    continue
                spot = (
                    chain.get("underlying_price") or 
                    chain.get("underlyingPrice") or 
                    chain.get("underlying_spot_price") or
                    chain.get("spot_price") or 
                    chain.get("spotPrice")
                )
                if spot is None:
                    # Try getting from the first entry of the chain
                    entries = chain.get("data") or chain.get("option_chain") or []
                    if entries and isinstance(entries, list) and len(entries) > 0:
                        first = entries[0]
                        if isinstance(first, dict):
                            spot = (
                                first.get("underlying_price") or 
                                first.get("underlyingPrice") or 
                                first.get("underlying_spot_price") or
                                first.get("spot_price") or 
                                first.get("spotPrice")
                            )

                if spot is None:
                    spot = await self._get_spot_price(api_base, instrument_map, underlying, underlying_key)
                ctx = {"expiry": expiry, "chain": chain, "spot": spot}
                cached_ctx[underlying_key] = ctx

            expiry = ctx.get("expiry")
            chain = ctx.get("chain")
            spot = ctx.get("spot")
            strike_val = self._resolve_target_strike(strike, step, spot)
            if strike_val is None:
                continue

            token_key, oi = self._pick_instrument(chain, strike_val, side)
            if token_key:
                inst["key"] = token_key
                inst["resolved_expiry"] = expiry
                inst["resolved_strike"] = strike_val
                if oi is not None:
                    inst["resolved_oi"] = oi

        return instruments

    def _prepare_instruments(self, options_cfg: Dict[str, Any], instruments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        strike_cfg = (options_cfg.get("strike_selection") or {}) if isinstance(options_cfg, dict) else {}
        mode = str(strike_cfg.get("mode", "atm_only")).lower()
        dynamic_range = int(strike_cfg.get("range", 2) or 2)
        max_total = int(strike_cfg.get("max_subscribed_instruments", 25) or 25)
        if max_total < 1:
            max_total = 25

        if mode == "dynamic":
            expanded = self._expand_dynamic_instruments(instruments, max(0, dynamic_range))
        elif mode == "fixed":
            expanded = [dict(item) for item in instruments if isinstance(item, dict)]
        else:
            # Default ATM only
            expanded = []
            seen = set()
            for item in instruments:
                if not isinstance(item, dict):
                    continue
                if str(item.get("strike", "ATM")).upper() != "ATM":
                    continue
                key = (item.get("underlying"), item.get("side"))
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(dict(item))

        # Deduplicate and cap.
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in expanded:
            marker = (
                item.get("underlying"),
                item.get("side"),
                str(item.get("strike")),
                int(item.get("step") or 0),
            )
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(item)
            if len(deduped) >= max_total:
                break
        return deduped

    def _expand_dynamic_instruments(self, instruments: List[Dict[str, Any]], dynamic_range: int) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        templates: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in instruments:
            if not isinstance(item, dict):
                continue
            underlying = item.get("underlying")
            side = item.get("side")
            if not underlying or side not in ("CE", "PE"):
                continue
            templates[(str(underlying), str(side))] = item

        offsets = list(range(-dynamic_range, dynamic_range + 1))
        for tpl in templates.values():
            for offset in offsets:
                strike_tag = "ATM" if offset == 0 else f"ATM{offset:+d}"
                generated = dict(tpl)
                generated["strike"] = strike_tag
                generated["offset"] = offset
                generated["key"] = None
                expanded.append(generated)
        return expanded

    def _resolve_target_strike(self, strike: Any, step: int, spot: Any) -> Optional[float]:
        if step <= 0:
            return None
        try:
            spot_v = float(spot) if spot is not None else None
        except Exception:
            spot_v = None

        strike_s = str(strike).upper() if strike is not None else "ATM"
        if strike_s.startswith("ATM"):
            if spot_v is None:
                return None
            atm = round(spot_v / step) * step
            if strike_s == "ATM":
                return float(atm)
            # ATM+1 / ATM-2 style.
            suffix = strike_s.replace("ATM", "", 1).strip()
            try:
                offset = int(suffix)
            except Exception:
                offset = 0
            return float(atm + (offset * step))

        try:
            return float(strike)
        except Exception:
            return None

    async def resolve_debug(self) -> Dict[str, Any]:
        options_cfg = self._config_engine.get_options_setting("options")
        md_cfg = options_cfg.get("market_data", {}) or {}
        instruments = md_cfg.get("option_instruments", []) or []
        instruments = self._prepare_instruments(options_cfg, instruments)

        market_cfg = self._config_engine.get_app_setting("market_data")
        upstox_cfg = market_cfg.get("upstox", {})
        api_base = upstox_cfg.get("api_base_url", "https://api.upstox.com")
        instrument_map: Dict[str, str] = upstox_cfg.get("instrument_keys", {}) or {}

        token = await ApiTokenService("upstox").get_token()
        if not token:
            return {"token_present": False, "instruments": instruments}

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        results = []
        for inst in instruments:
            if not isinstance(inst, dict):
                continue
            underlying = inst.get("underlying")
            step = int(inst.get("step") or 0)
            side = inst.get("side")
            strike = inst.get("strike")

            underlying_key = self._resolve_underlying_key(underlying, instrument_map)
            debug_item = {
                "underlying": underlying,
                "side": side,
                "strike": strike,
                "step": step,
                "underlying_key": underlying_key,
                "expiry": None,
                "atm_strike": None,
                "token_key": None,
                "contract_status": None,
                "chain_status": None,
            }

            if not underlying_key:
                results.append(debug_item)
                continue

            expiry, contract_status, contract_debug = await self._get_nearest_expiry_debug(api_base, headers, underlying_key)
            debug_item["contract_status"] = contract_status
            debug_item["expiry"] = expiry
            debug_item["contract_debug"] = contract_debug
            if not expiry:
                results.append(debug_item)
                continue

            chain, chain_status, chain_debug = await self._get_option_chain_debug(api_base, headers, underlying_key, expiry)
            debug_item["chain_status"] = chain_status
            debug_item["chain_debug"] = chain_debug
            if not chain:
                results.append(debug_item)
                continue

            spot = (
                chain.get("underlying_price") or 
                chain.get("underlyingPrice") or 
                chain.get("underlying_spot_price") or
                chain.get("spot_price") or 
                chain.get("spotPrice")
            )
            if spot is None:
                entries = chain.get("data") or chain.get("option_chain") or []
                if entries and isinstance(entries, list) and len(entries) > 0:
                    first = entries[0]
                    if isinstance(first, dict):
                        spot = (
                            first.get("underlying_price") or 
                            first.get("underlyingPrice") or 
                            first.get("underlying_spot_price") or
                            first.get("spot_price") or 
                            first.get("spotPrice")
                        )

            strike_val = self._resolve_target_strike(strike, step, spot)
            debug_item["atm_strike"] = strike_val

            token_key, _ = self._pick_instrument(chain, strike_val, side)
            debug_item["token_key"] = token_key
            results.append(debug_item)

        return {"token_present": True, "instruments": results}

    def _resolve_underlying_key(self, name: str, instrument_map: Dict[str, str]) -> Optional[str]:
        # Match common labels
        if name == "NIFTY 50":
            return instrument_map.get("NIFTY50") or instrument_map.get("NIFTY 50")
        if name == "NIFTY BANK":
            return instrument_map.get("BANKNIFTY") or instrument_map.get("NIFTY BANK")
        return instrument_map.get(name)

    async def _get_nearest_expiry(self, api_base: str, headers: Dict[str, str], underlying_key: str) -> Optional[str]:
        url = f"{api_base}/v2/option/contract"
        params = {"instrument_key": underlying_key}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        contracts = self._extract_contracts(payload)
        if not contracts:
            return None
        today = date.today()
        expiries: List[date] = []
        for item in contracts:
            exp = item.get("expiry_date") or item.get("expiryDate") or item.get("expiry")
            if not exp:
                continue
            try:
                exp_date = datetime.fromisoformat(exp).date()
            except Exception:
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                except Exception:
                    continue
            if exp_date >= today:
                expiries.append(exp_date)
        if not expiries:
            return None
        return min(expiries).isoformat()

    async def _get_nearest_expiry_debug(self, api_base: str, headers: Dict[str, str], underlying_key: str) -> tuple[Optional[str], Optional[int], Dict[str, Any]]:
        url = f"{api_base}/v2/option/contract"
        params = {"instrument_key": underlying_key}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None, resp.status_code, {"payload_keys": [], "contracts_len": 0}
        payload = resp.json()
        contracts = self._extract_contracts(payload)
        if not contracts:
            debug = {
                "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
                "contracts_len": 0,
                "sample": payload.get("data") if isinstance(payload, dict) else None,
            }
            return None, resp.status_code, debug
        today = date.today()
        expiries: List[date] = []
        for item in contracts:
            exp = item.get("expiry_date") or item.get("expiryDate") or item.get("expiry")
            if not exp:
                continue
            try:
                exp_date = datetime.fromisoformat(exp).date()
            except Exception:
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                except Exception:
                    continue
            if exp_date >= today:
                expiries.append(exp_date)
        if not expiries:
            return None, resp.status_code, {"contracts_len": len(contracts), "sample": contracts[0] if contracts else None}
        return min(expiries).isoformat(), resp.status_code, {"contracts_len": len(contracts), "sample": contracts[0] if contracts else None}

    async def _get_option_chain(self, api_base: str, headers: Dict[str, str], underlying_key: str, expiry: str) -> Optional[Dict[str, Any]]:
        url = f"{api_base}/v2/option/chain"
        params = {"instrument_key": underlying_key, "expiry_date": expiry}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"data": data}
        return None

    async def _get_option_chain_debug(self, api_base: str, headers: Dict[str, str], underlying_key: str, expiry: str) -> tuple[Optional[Dict[str, Any]], Optional[int], Dict[str, Any]]:
        url = f"{api_base}/v2/option/chain"
        params = {"instrument_key": underlying_key, "expiry_date": expiry}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None, resp.status_code, {"payload_keys": []}
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            chain_list = data.get("data") or data.get("option_chain") or []
            debug = {
                "payload_keys": list(data.keys()),
                "chain_len": len(chain_list) if isinstance(chain_list, list) else 0,
                "sample": chain_list[0] if isinstance(chain_list, list) and chain_list else None,
            }
            return data, resp.status_code, debug
        if isinstance(data, list):
            debug = {
                "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
                "chain_len": len(data),
                "sample": data[0] if data else None,
            }
            return {"data": data}, resp.status_code, debug
        return None, resp.status_code, {"payload_keys": list(payload.keys()) if isinstance(payload, dict) else []}

    async def _get_spot_price(
        self,
        api_base: str,
        instrument_map: Dict[str, str],
        underlying: str,
        underlying_key: str,
    ) -> Optional[float]:
        # Use UpstoxProvider to fetch current spot for underlying.
        provider = UpstoxProvider(
            api_base_url=api_base,
            api_key=None,
            api_secret=None,
            instrument_keys={underlying: underlying_key},
        )
        prices = await provider.get_current_prices([underlying])
        value = prices.get(underlying)
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _extract_contracts(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("contracts", "contract", "records", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(data, list):
            return data
        return []

    def _pick_instrument(self, chain: Dict[str, Any], strike: float, side: str) -> tuple[Optional[str], Optional[float]]:
        entries = chain.get("data") or chain.get("option_chain") or []
        if not isinstance(entries, list):
            return None, None
        for item in entries:
            strike_price = item.get("strike_price") or item.get("strikePrice")
            if strike_price is None:
                continue
            try:
                if float(strike_price) != float(strike):
                    continue
            except Exception:
                continue
            if side == "CE":
                call = item.get("call_options") or item.get("callOptions") or {}
                key = call.get("instrument_key") or call.get("instrumentKey")
                oi = call.get("market_data", {}).get("oi") if isinstance(call.get("market_data"), dict) else None
                return key, oi
            if side == "PE":
                put = item.get("put_options") or item.get("putOptions") or {}
                key = put.get("instrument_key") or put.get("instrumentKey")
                oi = put.get("market_data", {}).get("oi") if isinstance(put.get("market_data"), dict) else None
                return key, oi
        return None, None
