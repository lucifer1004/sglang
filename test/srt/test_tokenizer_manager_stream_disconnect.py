"""Unit tests for the streaming-disconnect abort fix in TokenizerManager.

This branch fixes a resource leak where, if a client closed its HTTP connection
mid-stream, the scheduler kept decoding because TokenizerManager never noticed
the disconnect on the streaming path.

The fix adds a `request.is_disconnected()` check in the streaming branch of
`TokenizerManager._wait_one_response`, right after a chunk arrives and before
yielding it to the client. On disconnect we call `self.abort_request(rid)` and
break out of the generator instead of yielding.

These tests pin that behavior:

  * positive case  - client is disconnected => abort fires, generator stops
                     before yielding the buffered chunk.
  * negative case  - client is connected    => abort does not fire and the
                     buffered chunk is yielded normally.

They run as plain `unittest` (no pytest required) and avoid spinning up a real
TokenizerManager by calling the unbound coroutine method against a
SimpleNamespace that only exposes `abort_request`.
"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def _streaming_state(rid: str = "rid-0"):
    """Build a minimal streaming ReqState with one buffered chunk and a set event."""
    obj = SimpleNamespace(rid=rid, stream=True, background=False)
    state = ReqState(
        out_list=[
            {
                "text": "hello",
                "output_ids": [1],
                "meta_info": {
                    "id": rid,
                    "finish_reason": None,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                },
            }
        ],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        created_time=time.time(),
    )
    state.event.set()
    return obj, state


def _fake_manager():
    """Return a (manager, aborted_calls) pair where abort_request appends to the list."""
    aborted = []
    manager = SimpleNamespace(
        abort_request=lambda rid="", abort_all=False: aborted.append((rid, abort_all))
    )
    return manager, aborted


class TestStreamingDisconnectAbort(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_aborts_and_stops_before_yielding(self):
        """Client disconnect must abort the request and stop the generator."""
        manager, aborted = _fake_manager()
        obj, state = _streaming_state("rid-disconnected")
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))

        gen = TokenizerManager._wait_one_response(manager, obj, state, request)

        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()

        self.assertEqual(
            aborted,
            [("rid-disconnected", False)],
            "abort_request must fire exactly once with the streaming rid",
        )
        request.is_disconnected.assert_awaited_once()

    async def test_connected_client_yields_chunk_without_aborting(self):
        """Negative control: a still-connected client must not be aborted."""
        manager, aborted = _fake_manager()
        obj, state = _streaming_state("rid-connected")
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

        gen = TokenizerManager._wait_one_response(manager, obj, state, request)
        out = await gen.__anext__()
        await gen.aclose()

        self.assertEqual(aborted, [], "abort_request must not fire while connected")
        self.assertEqual(out["meta_info"]["id"], "rid-connected")
        request.is_disconnected.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
