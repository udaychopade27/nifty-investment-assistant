#!/usr/bin/env python3
"""
Auto-map ETF symbols to Upstox instrument keys from the Upstox instruments CSV.

Usage:
  python scripts/upstox_map_etfs.py --instruments /path/to/instruments.csv
  python scripts/upstox_map_etfs.py --instruments /path/to/instruments.csv --write config/app.yml
  python scripts/upstox_map_etfs.py --instruments /path/to/instruments.csv --symbols NIFTYBEES,JUNIORBEES
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map ETF symbols to Upstox instrument keys.")
    parser.add_argument("--instruments", required=True, help="Path to Upstox instruments CSV/JSON")
    parser.add_argument("--download-url", default="", help="Optional URL to download instruments file")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to include (default: all in config)")
    parser.add_argument("--write", default="", help="Write mapping into config/app.yml")
    return parser.parse_args()


def load_symbols_from_config(config_path: Path) -> List[str]:
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    etfs = data.get("etfs") or data.get("etf_universe")
    # Not all configs have ETF list here; fall back to common list in config/etfs.yml
    if not etfs:
        etfs_path = config_path.parent / "etfs.yml"
        if etfs_path.exists():
            with etfs_path.open("r") as f:
                etf_data = yaml.safe_load(f) or {}
            etfs = etf_data.get("etfs", [])
    symbols = [e.get("symbol") for e in etfs or [] if isinstance(e, dict) and e.get("symbol")]
    indices = [e.get("underlying_index") for e in etfs or [] if isinstance(e, dict) and e.get("underlying_index")]
    base = sorted(set(s.upper() for s in symbols))
    # Include underlying indices used for dip logic
    for idx in indices:
        if idx:
            base.append(idx.upper())
    # Always include key indices for dip logic
    for idx in ("NIFTY50", "INDIA_VIX"):
        if idx not in base:
            base.append(idx)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for item in base:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def download_instruments(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        f.write(response.read())


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_symbol(value: str) -> str:
    return value.replace(" ", "").replace("_", "").upper()


def _prefer_exchange(existing_key: str, new_key: str) -> str:
    if not existing_key:
        return new_key
    # Prefer NSE over BSE if both exist
    if existing_key.startswith("BSE_") and new_key.startswith("NSE_"):
        return new_key
    return existing_key


def load_instrument_map(instruments_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    normalized: Dict[str, str] = {}
    if instruments_path.suffix in (".json", ".gz"):
        with _open_maybe_gzip(instruments_path) as f:
            data = json.load(f)
            for row in data:
                symbol = (
                    row.get("tradingsymbol")
                    or row.get("trading_symbol")
                    or ""
                ).strip().upper()
                name = (row.get("name") or "").strip().upper()
                instrument_key = (row.get("instrument_key") or "").strip()
                if not symbol or not instrument_key:
                    continue
                mapping[symbol] = _prefer_exchange(mapping.get(symbol, ""), instrument_key)
                norm_sym = _normalize_symbol(symbol)
                normalized[norm_sym] = _prefer_exchange(normalized.get(norm_sym, ""), instrument_key)
                if name:
                    norm_name = _normalize_symbol(name)
                    normalized[norm_name] = _prefer_exchange(normalized.get(norm_name, ""), instrument_key)
    else:
        with instruments_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Expected columns: instrument_key, tradingsymbol
            for row in reader:
                symbol = (
                    row.get("tradingsymbol")
                    or row.get("trading_symbol")
                    or ""
                ).strip().upper()
                name = (row.get("name") or "").strip().upper()
                instrument_key = (row.get("instrument_key") or "").strip()
                if not symbol or not instrument_key:
                    continue
                mapping[symbol] = _prefer_exchange(mapping.get(symbol, ""), instrument_key)
                norm_sym = _normalize_symbol(symbol)
                normalized[norm_sym] = _prefer_exchange(normalized.get(norm_sym, ""), instrument_key)
                if name:
                    norm_name = _normalize_symbol(name)
                    normalized[norm_name] = _prefer_exchange(normalized.get(norm_name, ""), instrument_key)
    # Overlay normalized keys for flexible matching
    mapping.update(normalized)
    return mapping


def search_index_keys(instruments_path: Path, patterns: List[str], limit: int = 10) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    compiled = [(_normalize_symbol(p), p) for p in patterns]
    if instruments_path.suffix in (".json", ".gz"):
        with _open_maybe_gzip(instruments_path) as f:
            data = json.load(f)
            for row in data:
                name = (row.get("name") or "").strip()
                key = (row.get("instrument_key") or "").strip()
                if not name or not key:
                    continue
                norm_name = _normalize_symbol(name)
                for norm_pat, original in compiled:
                    if norm_pat in norm_name:
                        results.setdefault(original, [])
                        if len(results[original]) < limit:
                            results[original].append(f"{name} -> {key}")
    else:
        with instruments_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                key = (row.get("instrument_key") or "").strip()
                if not name or not key:
                    continue
                norm_name = _normalize_symbol(name)
                for norm_pat, original in compiled:
                    if norm_pat in norm_name:
                        results.setdefault(original, [])
                        if len(results[original]) < limit:
                            results[original].append(f"{name} -> {key}")
    return results


def update_config(config_path: Path, symbol_map: Dict[str, str]) -> None:
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    market_data = data.setdefault("market_data", {})
    upstox = market_data.setdefault("upstox", {})
    keys = upstox.setdefault("instrument_keys", {})
    for symbol, key in symbol_map.items():
        keys[symbol] = key
    with config_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    args = parse_args()
    instruments_path = Path(args.instruments)
    if args.download_url:
        try:
            download_instruments(args.download_url, instruments_path)
        except Exception as exc:
            print(f"ERROR: failed to download instruments file: {exc}")
            return 1
    if not instruments_path.exists():
        print(f"ERROR: instruments file not found: {instruments_path}")
        return 1

    config_path = Path(args.write) if args.write else Path("config/app.yml")
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else load_symbols_from_config(Path("config/app.yml"))
    )

    instrument_map = load_instrument_map(instruments_path)

    resolved = {}
    for sym in symbols:
        key = instrument_map.get(sym)
        if not key:
            key = instrument_map.get(_normalize_symbol(sym))
        resolved[sym] = key
    missing = [sym for sym, key in resolved.items() if not key]
    found = {sym: key for sym, key in resolved.items() if key}

    print("Resolved instrument keys (ETFs + indices):")
    for sym, key in found.items():
        print(f"  {sym}: {key}")

    if missing:
        print("\nMissing symbols:")
        for sym in missing:
            print(f"  {sym}")

    if args.write:
        update_config(config_path, found)
        print(f"\nUpdated {config_path} with {len(found)} keys.")
    else:
        print("\nTip: use --write config/app.yml to update the config file.")

    if missing:
        print("\nSearch suggestions for missing indices:")
        suggestions = search_index_keys(instruments_path, missing)
        for sym, matches in suggestions.items():
            if not matches:
                continue
            print(f"\n{sym}:")
            for item in matches:
                print(f"  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
