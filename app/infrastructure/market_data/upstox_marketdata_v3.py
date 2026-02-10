"""Upstox Market Data Feed V3 protobuf decoding (dynamic schema)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, List, Optional, Tuple

from google.protobuf import descriptor_pb2, descriptor_pool, json_format
try:
    from google.protobuf.message_factory import GetMessageClass  # type: ignore
except Exception:  # pragma: no cover
    GetMessageClass = None  # type: ignore

_PACKAGE = "com.upstox.marketdatafeederv3udapi.rpc.proto"

_FEED_RESPONSE_CLASS = None


def _add_enum(file_desc: descriptor_pb2.FileDescriptorProto, name: str, values: List[str]) -> None:
    enum = file_desc.enum_type.add()
    enum.name = name
    for idx, value_name in enumerate(values):
        value = enum.value.add()
        value.name = value_name
        value.number = idx


def _add_field(
    msg: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    label: int = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    type_name: Optional[str] = None,
    oneof_index: Optional[int] = None,
) -> None:
    field = msg.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name
    if oneof_index is not None:
        field.oneof_index = oneof_index


def _build_descriptor() -> descriptor_pb2.FileDescriptorProto:
    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "MarketDataFeed.proto"
    file_desc.package = _PACKAGE
    file_desc.syntax = "proto3"

    # Enums
    _add_enum(file_desc, "Type", ["initial_feed", "live_feed", "market_info"])
    _add_enum(file_desc, "RequestMode", ["ltpc", "full_d5", "option_greeks", "full_d30"])
    _add_enum(
        file_desc,
        "MarketStatus",
        [
            "PRE_OPEN_START",
            "PRE_OPEN_END",
            "NORMAL_OPEN",
            "NORMAL_CLOSE",
            "CLOSING_START",
            "CLOSING_END",
        ],
    )

    # Messages
    ltpc = file_desc.message_type.add()
    ltpc.name = "LTPC"
    _add_field(ltpc, "ltp", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(ltpc, "ltt", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(ltpc, "ltq", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(ltpc, "cp", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    quote = file_desc.message_type.add()
    quote.name = "Quote"
    _add_field(quote, "bidQ", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(quote, "bidP", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(quote, "askQ", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(quote, "askP", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    market_level = file_desc.message_type.add()
    market_level.name = "MarketLevel"
    _add_field(
        market_level,
        "bidAskQuote",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f".{_PACKAGE}.Quote",
    )

    ohlc = file_desc.message_type.add()
    ohlc.name = "OHLC"
    _add_field(ohlc, "interval", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(ohlc, "open", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(ohlc, "high", 3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(ohlc, "low", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(ohlc, "close", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(ohlc, "vol", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(ohlc, "ts", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)

    market_ohlc = file_desc.message_type.add()
    market_ohlc.name = "MarketOHLC"
    _add_field(
        market_ohlc,
        "ohlc",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f".{_PACKAGE}.OHLC",
    )

    option_greeks = file_desc.message_type.add()
    option_greeks.name = "OptionGreeks"
    _add_field(option_greeks, "delta", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(option_greeks, "theta", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(option_greeks, "gamma", 3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(option_greeks, "vega", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(option_greeks, "rho", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    market_full = file_desc.message_type.add()
    market_full.name = "MarketFullFeed"
    _add_field(market_full, "ltpc", 1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.LTPC")
    _add_field(
        market_full,
        "marketLevel",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.MarketLevel",
    )
    _add_field(
        market_full,
        "optionGreeks",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.OptionGreeks",
    )
    _add_field(
        market_full,
        "marketOHLC",
        4,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.MarketOHLC",
    )
    _add_field(market_full, "atp", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(market_full, "vtt", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(market_full, "oi", 7, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(market_full, "iv", 8, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(market_full, "tbq", 9, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(market_full, "tsq", 10, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    index_full = file_desc.message_type.add()
    index_full.name = "IndexFullFeed"
    _add_field(index_full, "ltpc", 1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.LTPC")
    _add_field(
        index_full,
        "marketOHLC",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.MarketOHLC",
    )

    full_feed = file_desc.message_type.add()
    full_feed.name = "FullFeed"
    full_feed.oneof_decl.add().name = "FullFeedUnion"
    _add_field(
        full_feed,
        "marketFF",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.MarketFullFeed",
        oneof_index=0,
    )
    _add_field(
        full_feed,
        "indexFF",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.IndexFullFeed",
        oneof_index=0,
    )

    first_level = file_desc.message_type.add()
    first_level.name = "FirstLevelWithGreeks"
    _add_field(first_level, "ltpc", 1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.LTPC")
    _add_field(first_level, "firstDepth", 2, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.Quote")
    _add_field(first_level, "optionGreeks", 3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.OptionGreeks")
    _add_field(first_level, "vtt", 4, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(first_level, "oi", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(first_level, "iv", 6, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    feed = file_desc.message_type.add()
    feed.name = "Feed"
    feed.oneof_decl.add().name = "FeedUnion"
    _add_field(feed, "ltpc", 1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.LTPC", oneof_index=0)
    _add_field(feed, "fullFeed", 2, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.FullFeed", oneof_index=0)
    _add_field(
        feed,
        "firstLevelWithGreeks",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.FirstLevelWithGreeks",
        oneof_index=0,
    )
    _add_field(feed, "requestMode", 4, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, type_name=f".{_PACKAGE}.RequestMode")

    market_info = file_desc.message_type.add()
    market_info.name = "MarketInfo"
    seg_entry = market_info.nested_type.add()
    seg_entry.name = "SegmentStatusEntry"
    seg_entry.options.map_entry = True
    _add_field(seg_entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(
        seg_entry,
        "value",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        type_name=f".{_PACKAGE}.MarketStatus",
    )
    _add_field(
        market_info,
        "segmentStatus",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f".{_PACKAGE}.MarketInfo.SegmentStatusEntry",
    )

    feed_response = file_desc.message_type.add()
    feed_response.name = "FeedResponse"
    feeds_entry = feed_response.nested_type.add()
    feeds_entry.name = "FeedsEntry"
    feeds_entry.options.map_entry = True
    _add_field(feeds_entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(
        feeds_entry,
        "value",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=f".{_PACKAGE}.Feed",
    )
    _add_field(feed_response, "type", 1, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, type_name=f".{_PACKAGE}.Type")
    _add_field(
        feed_response,
        "feeds",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f".{_PACKAGE}.FeedResponse.FeedsEntry",
    )
    _add_field(feed_response, "currentTs", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(feed_response, "marketInfo", 4, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f".{_PACKAGE}.MarketInfo")

    return file_desc


def _get_feed_response_class():
    global _FEED_RESPONSE_CLASS
    if _FEED_RESPONSE_CLASS is not None:
        return _FEED_RESPONSE_CLASS

    file_desc = _build_descriptor()
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(file_desc.SerializeToString())
    descriptor = pool.FindMessageTypeByName(f"{_PACKAGE}.FeedResponse")
    if GetMessageClass is not None:
        _FEED_RESPONSE_CLASS = GetMessageClass(descriptor)
    else:
        _FEED_RESPONSE_CLASS = pool.GetMessageClass(descriptor.full_name)
    return _FEED_RESPONSE_CLASS


def decode_feed_response(raw: bytes) -> List[dict]:
    """Decode v3 protobuf feed response into event dicts."""
    message_cls = _get_feed_response_class()
    msg = message_cls()
    msg.ParseFromString(raw)

    events: List[dict] = []
    feeds = getattr(msg, "feeds", {})
    for key, feed in feeds.items():
        ltpc = None
        if feed.HasField("ltpc"):
            ltpc = feed.ltpc
        elif feed.HasField("fullFeed"):
            full_feed = feed.fullFeed
            if full_feed.HasField("marketFF"):
                ltpc = full_feed.marketFF.ltpc
            elif full_feed.HasField("indexFF"):
                ltpc = full_feed.indexFF.ltpc
        elif feed.HasField("firstLevelWithGreeks"):
            ltpc = feed.firstLevelWithGreeks.ltpc

        if ltpc is None:
            continue
        ltp = getattr(ltpc, "ltp", None)
        ltt = getattr(ltpc, "ltt", None)
        oi = None
        iv = None
        delta = None
        vtt = None
        bid = None
        ask = None
        try:
            if feed.HasField("fullFeed"):
                full_feed = feed.fullFeed
                if full_feed.HasField("marketFF"):
                    oi = getattr(full_feed.marketFF, "oi", None)
                    iv = getattr(full_feed.marketFF, "iv", None)
                    vtt = getattr(full_feed.marketFF, "vtt", None)
                    try:
                        if full_feed.marketFF.HasField("optionGreeks"):
                            delta = getattr(full_feed.marketFF.optionGreeks, "delta", None)
                    except Exception:
                        delta = None
                    try:
                        quotes = getattr(full_feed.marketFF.marketLevel, "bidAskQuote", None)
                        if quotes:
                            first = quotes[0]
                            bid = getattr(first, "bidP", None)
                            ask = getattr(first, "askP", None)
                    except Exception:
                        bid = None
                        ask = None
                elif full_feed.HasField("indexFF"):
                    oi = None
        except Exception:
            oi = None
            iv = None
            delta = None
            vtt = None
            bid = None
            ask = None
        if ltp is None:
            continue
        ts = _ts_from_ltt(ltt)
        events.append(
            {
                "instrument_key": key,
                "ltp": Decimal(str(ltp)),
                "ts": ts,
                "oi": float(oi) if oi is not None else None,
                "iv": float(iv) if iv is not None else None,
                "delta": float(delta) if delta is not None else None,
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "volume": float(vtt) if vtt is not None else None,
            }
        )

    if events:
        return events

    # Fallback: convert to dict and search for ltp/ltt regardless of schema drift.
    try:
        payload = json_format.MessageToDict(msg, preserving_proto_field_name=True)
    except Exception:
        return events

    feeds = payload.get("feeds") if isinstance(payload, dict) else None
    if not isinstance(feeds, dict):
        return events

    for key, feed in feeds.items():
        if not isinstance(feed, dict):
            continue
        ltp, ltt = _find_ltp_ltt(feed)
        oi = _find_oi(feed)
        iv = _find_iv(feed)
        delta = _find_delta(feed)
        vtt = _find_vtt(feed)
        bid, ask = _find_bid_ask(feed)
        if ltp is None:
            continue
        events.append(
            {
                "instrument_key": key,
                "ltp": Decimal(str(ltp)),
                "ts": _ts_from_ltt(ltt),
                "oi": oi,
                "iv": iv,
                "delta": delta,
                "bid": bid,
                "ask": ask,
                "volume": vtt,
            }
        )

    return events


def _ts_from_ltt(ltt: Optional[int]) -> datetime:
    if not ltt:
        return datetime.now(tz=timezone.utc)
    # Heuristic: ltt in ms if large
    if ltt > 10_000_000_000:
        return datetime.fromtimestamp(ltt / 1000.0, tz=timezone.utc)
    return datetime.fromtimestamp(ltt, tz=timezone.utc)


def _find_ltp_ltt(payload: object) -> Tuple[Optional[float], Optional[int]]:
    """Depth-first search for ltp and ltt in a nested dict/list."""
    ltp = None
    ltt = None
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "ltp" in node and ltp is None:
                try:
                    ltp = float(node["ltp"])
                except Exception:
                    pass
            if "ltt" in node and ltt is None:
                try:
                    ltt = int(node["ltt"])
                except Exception:
                    pass
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
        if ltp is not None and ltt is not None:
            break
    return ltp, ltt


def _find_oi(payload: object) -> Optional[float]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "oi" in node:
                try:
                    return float(node["oi"])
                except Exception:
                    return None
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
    return None


def _find_vtt(payload: object) -> Optional[float]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "vtt" in node:
                try:
                    return float(node["vtt"])
                except Exception:
                    return None
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
    return None


def _find_iv(payload: object) -> Optional[float]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "iv" in node:
                try:
                    return float(node["iv"])
                except Exception:
                    return None
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
    return None


def _find_delta(payload: object) -> Optional[float]:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "delta" in node:
                try:
                    return float(node["delta"])
                except Exception:
                    return None
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
    return None


def _find_bid_ask(payload: object) -> Tuple[Optional[float], Optional[float]]:
    bid = None
    ask = None
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if bid is None and "bidP" in node:
                try:
                    bid = float(node["bidP"])
                except Exception:
                    bid = None
            if ask is None and "askP" in node:
                try:
                    ask = float(node["askP"])
                except Exception:
                    ask = None
            for value in node.values():
                stack.append(value)
        elif isinstance(node, list):
            for item in node:
                stack.append(item)
        if bid is not None and ask is not None:
            break
    return bid, ask


def debug_feed_summary(raw: bytes) -> dict:
    """Best-effort summary of decoded feed for diagnostics."""
    message_cls = _get_feed_response_class()
    msg = message_cls()
    msg.ParseFromString(raw)
    try:
        payload = json_format.MessageToDict(msg, preserving_proto_field_name=True)
    except Exception as exc:
        return {"error": str(exc)}
    feeds = payload.get("feeds") if isinstance(payload, dict) else None
    keys = list(feeds.keys())[:5] if isinstance(feeds, dict) else []
    sample = feeds.get(keys[0]) if isinstance(feeds, dict) and keys else None
    return {
        "type": payload.get("type") if isinstance(payload, dict) else None,
        "marketInfo": payload.get("marketInfo") if isinstance(payload, dict) else None,
        "keys": keys,
        "sample": sample,
        "top_keys": list(payload.keys()) if isinstance(payload, dict) else [],
    }
