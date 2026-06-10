"""Test that LLM errors propagate through session.run() → RunResult,
including the full e2e path through SessionHost → RemoteSession."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from livekit.agents import APIStatusError
from livekit.agents.stt import STTError
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.remote_session import (
    RemoteSession,
    SessionHost,
    SessionTransport,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class FailingLLM(FakeLLM):
    """A FakeLLM that raises a retryable API error, going through the retry loop."""

    def chat(self, **kwargs):
        raise APIStatusError(
            "object cannot be found",
            status_code=401,
            retryable=True,
        )


class PairedTransport(SessionTransport):
    def __init__(self) -> None:
        self._inbox: asyncio.Queue[agent_pb.AgentSessionMessage] = asyncio.Queue()
        self._peer: PairedTransport | None = None
        self._closed = False

    @classmethod
    def create_pair(cls) -> tuple[PairedTransport, PairedTransport]:
        a, b = cls(), cls()
        a._peer = b
        b._peer = a
        return a, b

    async def start(self) -> None:
        pass

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._peer and not self._peer._closed:
            self._peer._inbox.put_nowait(msg)

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_run_propagates_llm_error_no_retry():
    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0))
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(APIStatusError):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_propagates_llm_error_with_retry():
    session = AgentSession(
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01)
        )
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(APIStatusError):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_input_error_e2e_through_remote_session():
    """Full e2e: RemoteSession → SessionHost → AgentSession with failing LLM.

    Verifies that an LLM 401 error propagates all the way back to the
    RemoteSession.run_input() caller as a RuntimeError.
    """
    host_transport, client_transport = PairedTransport.create_pair()

    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0))
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    host = SessionHost(host_transport)
    host.register_session(session)

    await session.start(agent=agent)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    with pytest.raises(RuntimeError, match="failed"):
        await client.run("order a big mac", timeout=10.0)

    await client.aclose()
    await host.aclose()
    await session.aclose()


@pytest.mark.asyncio
async def test_stt_error_respects_max_unrecoverable_errors():
    """STT unrecoverable errors must honour max_unrecoverable_errors,
    matching the tolerance already applied to LLM and TTS errors."""
    max_errors = 2
    session = AgentSession(conn_options=SessionConnectOptions(max_unrecoverable_errors=max_errors))
    agent = Agent(instructions="test", llm=FailingLLM())
    await session.start(agent=agent)

    def make_stt_error() -> STTError:
        return STTError(
            timestamp=time.time(),
            label="test-stt",
            error=RuntimeError("stt failed"),
            recoverable=False,
        )

    close_events: list[object] = []
    session.on("close", close_events.append)

    # Errors within the tolerance window must not close the session.
    for _ in range(max_errors):
        session._on_error(make_stt_error())
        await asyncio.sleep(0)
    assert not close_events, "session should remain open within tolerance"

    # One more error crosses the threshold — session must close.
    session._on_error(make_stt_error())
    # Give the closing task a few event-loop ticks to emit the close event.
    for _ in range(10):
        await asyncio.sleep(0)
    assert close_events, "session should have closed after exceeding tolerance"

    await session.aclose()
