import pytest
from datetime import datetime, timezone

from app.realtime.signal_queue import SignalQueue, SignalWorker, SignalEvent


@pytest.mark.asyncio
async def test_signal_worker_handles_event():
    queue = SignalQueue()
    seen = []

    async def handler(event: SignalEvent):
        seen.append(event.event_type)

    worker = SignalWorker(queue, handler)
    worker.start()

    await queue.publish(
        SignalEvent(
            event_type="bar_close",
            symbol="ABC",
            ts=datetime.now(tz=timezone.utc),
            payload={"open": 100, "close": 101},
        )
    )

    # allow worker to process
    await queue.join()
    await worker.stop()

    assert seen == ["bar_close"]
