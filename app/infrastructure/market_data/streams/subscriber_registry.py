"""Registry for market data subscribers."""
import inspect
from typing import Callable, Dict, List, Any, Awaitable


class SubscriberRegistry:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], Any]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, event: Any) -> None:
        for handler in self._subscribers.get(topic, []):
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    def count(self, topic: str) -> int:
        return len(self._subscribers.get(topic, []))
