### Quick Start: SQLAlchemy Session with Existing Engine

Source: https://openai.github.io/openai-agents-python/sessions/sqlalchemy_session

Shows how to create an `SQLAlchemySession` using an existing SQLAlchemy asynchronous engine (PostgreSQL example). This is useful for integrating with applications that already have a database engine configured. The example includes engine creation, session setup, task execution, and engine disposal.

```python
import asyncio
from agents import Agent, Runner
from agents.extensions.memory import SQLAlchemySession
from sqlalchemy.ext.asyncio import create_async_engine

async def main():
    # Create your database engine
    engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")

    agent = Agent("Assistant")
    session = SQLAlchemySession(
        "user-456",
        engine=engine,
        create_tables=True
    )

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

    # Clean up
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### Create Project and Virtual Environment (Bash)

Source: https://openai.github.io/openai-agents-python/quickstart

These commands set up a new project directory and create a Python virtual environment named '.venv'. This is a one-time setup for your project.

```bash
mkdir my_project
cd my_project
python -m venv .venv
```

--------------------------------

### Activate Virtual Environment (Bash)

Source: https://openai.github.io/openai-agents-python/quickstart

This command activates the Python virtual environment. You need to run this every time you start a new terminal session to use the installed packages.

```bash
source .venv/bin/activate
```

--------------------------------

### Configure Realtime Runner (Python)

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

Sets up the RealtimeRunner with the starting agent and configuration for model settings, including model name, voice, audio formats, transcription, and turn detection. Dependencies: RealtimeRunner class, agent instance.

```python
runner = RealtimeRunner(
    starting_agent=agent,
    config={
        "model_settings": {
            "model_name": "gpt-realtime",
            "voice": "ash",
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            "turn_detection": {"type": "semantic_vad", "interrupt_response": True},
        }
    }
)
```

--------------------------------

### Complete Voice Pipeline Example

Source: https://openai.github.io/openai-agents-python/voice/quickstart

Combines all the previous steps into a single, runnable script. This includes setting up agents, initializing the voice pipeline, generating sample audio input, running the pipeline, and streaming the audio output. It requires `asyncio` to manage the asynchronous operations.

```python
import asyncio
import random

import numpy as np
import sounddevice as sd

from agents import (
    Agent,
    function_tool,
    set_tracing_disabled,
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-5.2",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-5.2",
    handoffs=[spanish_agent],
    tools=[get_weather],
)


async def main():
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### Install OpenAI Agents with Voice Dependencies

Source: https://openai.github.io/openai-agents-python/voice/quickstart

Installs the OpenAI Agents SDK along with optional voice-related dependencies. This is a prerequisite for using voice features.

```shell
pip install 'openai-agents[voice]'
```

--------------------------------

### Realtime Agent and Runner Example in Python

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

This Python code demonstrates how to set up and run a real-time conversational agent. It initializes a RealtimeAgent with specific instructions and a RealtimeRunner with detailed model and audio configurations. The code then processes various events emitted during the agent's session, such as agent start/end, tool usage, and audio events. It relies on the asyncio library for asynchronous operations and the agents.realtime module.

```python
import asyncio
from agents.realtime import RealtimeAgent, RealtimeRunner

def _truncate_str(s: str, max_length: int) -> str:
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s

async def main():
    # Create the agent
    agent = RealtimeAgent(
        name="Assistant",
        instructions="You are a helpful voice assistant. Keep responses brief and conversational.",
    )
    # Set up the runner with configuration
    runner = RealtimeRunner(
        starting_agent=agent,
        config={
            "model_settings": {
                "model_name": "gpt-realtime",
                "voice": "ash",
                "modalities": ["audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                "turn_detection": {"type": "semantic_vad", "interrupt_response": True},
            }
        },
    )
    # Start the session
    session = await runner.run()

    async with session:
        print("Session started! The agent will stream audio responses in real-time.")
        # Process events
        async for event in session:
            try:
                if event.type == "agent_start":
                    print(f"Agent started: {event.agent.name}")
                elif event.type == "agent_end":
                    print(f"Agent ended: {event.agent.name}")
                elif event.type == "handoff":
                    print(f"Handoff from {event.from_agent.name} to {event.to_agent.name}")
                elif event.type == "tool_start":
                    print(f"Tool started: {event.tool.name}")
                elif event.type == "tool_end":
                    print(f"Tool ended: {event.tool.name}; output: {event.output}")
                elif event.type == "audio_end":
                    print("Audio ended")
                elif event.type == "audio":
                    # Enqueue audio for callback-based playback with metadata
                    # Non-blocking put; queue is unbounded, so drops won’t occur.
                    pass
                elif event.type == "audio_interrupted":
                    print("Audio interrupted")
                    # Begin graceful fade + flush in the audio callback and rebuild jitter buffer.
                elif event.type == "error":
                    print(f"Error: {event.error}")
                elif event.type == "history_updated":
                    pass  # Skip these frequent events
                elif event.type == "history_added":
                    pass  # Skip these frequent events
                elif event.type == "raw_model_event":
                    print(f"Raw model event: {_truncate_str(str(event.data), 200)}")
                else:
                    print(f"Unknown event type: {event.type}")
            except Exception as e:
                print(f"Error processing event: {_truncate_str(str(e), 200)}")

if __name__ == "__main__":
    # Run the session
    asyncio.run(main())

```

--------------------------------

### RealtimeRunner run method example (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/runner

Starts and returns a realtime session, enabling bidirectional communication with the model. The example demonstrates initiating a session and sending a message, followed by iterating through events. Requires RealtimeSession.

```Python
    async def run(
        self, *, context: TContext | None = None, model_config: RealtimeModelConfig | None = None
    ) -> RealtimeSession:
        """Start and returns a realtime session.

        Returns:
            RealtimeSession: A session object that allows bidirectional communication with the
            realtime model.

        Example:
            ```python
            runner = RealtimeRunner(agent)
            async with await runner.run() as session:
                await session.send_message("Hello")
                async for event in session:
                    print(event)
            ```
        """
        # Create and return the connection
        session = RealtimeSession(
            model=self._model,
            agent=self._starting_agent,
            context=context,
            model_config=model_config,
            run_config=self._config,
        )

        return session

```

--------------------------------

### Run and Process Realtime Agent Session (Python)

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

Starts a realtime agent session using the configured runner and asynchronously iterates through events. It handles various event types like agent start/end, handoffs, tool usage, audio events, and errors. Includes a helper function for truncating long strings. Dependencies: asyncio, RealtimeRunner, RealtimeAgent.

```python
# Start the session
session = await runner.run()

async with session:
    print("Session started! The agent will stream audio responses in real-time.")
    # Process events
    async for event in session:
        try:
            if event.type == "agent_start":
                print(f"Agent started: {event.agent.name}")
            elif event.type == "agent_end":
                print(f"Agent ended: {event.agent.name}")
            elif event.type == "handoff":
                print(f"Handoff from {event.from_agent.name} to {event.to_agent.name}")
            elif event.type == "tool_start":
                print(f"Tool started: {event.tool.name}")
            elif event.type == "tool_end":
                print(f"Tool ended: {event.tool.name}; output: {event.output}")
            elif event.type == "audio_end":
                print("Audio ended")
            elif event.type == "audio":
                # Enqueue audio for callback-based playback with metadata
                # Non-blocking put; queue is unbounded, so drops won’t occur.
                pass
            elif event.type == "audio_interrupted":
                print("Audio interrupted")
                # Begin graceful fade + flush in the audio callback and rebuild jitter buffer.
            elif event.type == "error":
                print(f"Error: {event.error}")
            elif event.type == "history_updated":
                pass  # Skip these frequent events
            elif event.type == "history_added":
                pass  # Skip these frequent events
            elif event.type == "raw_model_event":
                print(f"Raw model event: {_truncate_str(str(event.data), 200)}")
            else:
                print(f"Unknown event type: {event.type}")
        except Exception as e:
            print(f"Error processing event: {_truncate_str(str(e), 200)}")

def _truncate_str(s: str, max_length: int) -> str:
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s

```

--------------------------------

### Quick Start: Advanced SQLite Session with Agent

Source: https://openai.github.io/openai-agents-python/sessions/advanced_sqlite_session

Demonstrates how to initialize an Agent and an AdvancedSQLiteSession, run a conversation turn, store usage data, and continue the conversation. This example highlights the basic workflow for using the advanced session features.

```python
from agents import Agent, Runner
from agents.extensions.memory import AdvancedSQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create an advanced session
session = AdvancedSQLiteSession(
    session_id="conversation_123",
    db_path="conversations.db",
    create_tables=True
)

# First conversation turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# IMPORTANT: Store usage data
await session.store_run_usage(result)

# Continue conversation
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"
await session.store_run_usage(result)

```

--------------------------------

### Run Complete Agent Workflow with Handoffs and Input Guardrail (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

This Python script demonstrates a complete agent workflow. It defines multiple agents (Guardrail check, Math Tutor, History Tutor, Triage Agent) and orchestrates their interaction using handoffs and an input guardrail. The `homework_guardrail` function checks user input, and the `triage_agent` routes queries to appropriate specialist agents. The `main` function provides examples of running the workflow with different inputs.

```python
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    # Example 1: History question
    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

    # Example 2: General/philosophical question
    try:
        result = await Runner.run(triage_agent, "What is the meaning of life?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### Quick Start: SQLAlchemy Session from Database URL

Source: https://openai.github.io/openai-agents-python/sessions/sqlalchemy_session

Demonstrates creating an `SQLAlchemySession` using a database URL (SQLite in-memory example). It initializes an agent and runner, then executes a task using the defined session. Includes necessary imports and asynchronous execution.

```python
import asyncio
from agents import Agent, Runner
from agents.extensions.memory import SQLAlchemySession

async def main():
    agent = Agent("Assistant")

    # Create session using database URL
    session = SQLAlchemySession.from_url(
        "user-123",
        url="sqlite+aiosqlite:///:memory:",
        create_tables=True
    )

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### Hello World Example with OpenAI Agents SDK (Python)

Source: https://openai.github.io/openai-agents-python/index

A 'Hello World' example demonstrating the basic usage of the OpenAI Agents SDK in Python. It initializes an Agent and uses the Runner to execute a task, printing the final output. Requires the OPENAI_API_KEY environment variable to be set.

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

--------------------------------

### RealtimeRunner Initialization and Run Method (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/runner

This Python code defines the RealtimeRunner class, responsible for managing real-time agent sessions. It includes the `__init__` method for setting up the runner with a starting agent, an optional model, and configuration, and the `run` method to initiate a session. The `run` method returns a `RealtimeSession` object for bidirectional communication. The example demonstrates how to instantiate the runner and start a session.

```python
class RealtimeRunner:
    """A `RealtimeRunner` is the equivalent of `Runner` for realtime agents. It automatically
    handles multiple turns by maintaining a persistent connection with the underlying model
    layer.

    The session manages the local history copy, executes tools, runs guardrails and facilitates
    handoffs between agents.

    Since this code runs on your server, it uses WebSockets by default. You can optionally create
    your own custom model layer by implementing the `RealtimeModel` interface.
    """

    def __init__(
        self,
        starting_agent: RealtimeAgent,
        *,
        model: RealtimeModel | None = None,
        config: RealtimeRunConfig | None = None,
    ) -> None:
        """Initialize the realtime runner.

        Args:
            starting_agent: The agent to start the session with.
            context: The context to use for the session.
            model: The model to use. If not provided, will use a default OpenAI realtime model.
            config: Override parameters to use for the entire run.
        """
        self._starting_agent = starting_agent
        self._config = config
        self._model = model or OpenAIRealtimeWebSocketModel()

    async def run(
        self, *, context: TContext | None = None, model_config: RealtimeModelConfig | None = None
    ) -> RealtimeSession:
        """Start and returns a realtime session.

        Returns:
            RealtimeSession: A session object that allows bidirectional communication with the
            realtime model.

        Example:
            ```python
            runner = RealtimeRunner(agent)
            async with await runner.run() as session:
                await session.send_message("Hello")
                async for event in session:
                    print(event)
            ```
        """
        # Create and return the connection
        session = RealtimeSession(
            model=self._model,
            agent=self._starting_agent,
            context=context,
            model_config=model_config,
            run_config=self._config,
        )

        return session

```

--------------------------------

### Basic Agent Configuration

Source: https://openai.github.io/openai-agents-python/agents

Demonstrates the basic configuration of an Agent, including its name, instructions, the language model to use, and any associated tools. The example shows how to define a simple function tool for getting weather information and include it in the agent's configuration.

```python
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gpt-5-nano",
    tools=[get_weather],
)

```

--------------------------------

### Initialization: AdvancedSQLiteSession

Source: https://openai.github.io/openai-agents-python/sessions/advanced_sqlite_session

Shows different ways to initialize an AdvancedSQLiteSession, including basic setup, with persistent storage via `db_path`, and with a custom logger instance. All examples ensure the necessary tables are created.

```python
from agents.extensions.memory import AdvancedSQLiteSession

# Basic initialization
session = AdvancedSQLiteSession(
    session_id="my_conversation",
    create_tables=True  # Auto-create advanced tables
)

# With persistent storage
session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="path/to/conversations.db",
    create_tables=True
)

# With custom logger
import logging
logger = logging.getLogger("my_app")
session = AdvancedSQLiteSession(
    session_id="session_456",
    create_tables=True,
    logger=logger
)

```

--------------------------------

### Python Trace Context Manager Example

Source: https://openai.github.io/openai-agents-python/ref/tracing/traces

Provides an example of using the `trace` context manager in Python for basic workflow tracing. It shows how to wrap a sequence of operations within a `with trace(...)` block to automatically manage the start and end of a trace, ensuring reliable cleanup and automatic span finalization.

```python
from src.agents.tracing.traces import trace

# Assuming Runner is defined elsewhere
# async def Runner.run(agent, data):
#     ...

async def basic_trace_example(order_data):
    with trace("Order Processing") as t:
        validation_result = await Runner.run(validator, order_data)
        if validation_result.approved:
            await Runner.run(processor, order_data)
    # Trace automatically finishes here
```

--------------------------------

### Full Example: Agent with LiteLLM Model and Tools

Source: https://openai.github.io/openai-agents-python/models/litellm

This Python script demonstrates setting up an agent that utilizes LiteLLM for model interaction. It defines a tool, creates an agent with a LiteLLM model (requiring model name and API key input), and runs a simple query. The example handles dynamic input for model and API key via command-line arguments or prompts.

```python
from __future__ import annotations

import asyncio

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main(model: str, api_key: str):
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(model=model, api_key=api_key),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    # First try to get model/api key from args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--api-key", type=str, required=False)
    args = parser.parse_args()

    model = args.model
    if not model:
        model = input("Enter a model name for Litellm: ")

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter an API key for Litellm: ")

    asyncio.run(main(model, api_key))
```

--------------------------------

### RealtimeRunner Initialization and Run Method

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/runner

Initializes the RealtimeRunner with a starting agent and an optional model or configuration. The `run` method starts a real-time session, enabling bidirectional communication with the model. It can be used asynchronously with a context manager.

```python
class RealtimeRunner:
    """A `RealtimeRunner` is the equivalent of `Runner` for realtime agents. It automatically
    handles multiple turns by maintaining a persistent connection with the underlying model
    layer.

    The session manages the local history copy, executes tools, runs guardrails and facilitates
    handoffs between agents.

    Since this code runs on your server, it uses WebSockets by default. You can optionally create
    your own custom model layer by implementing the `RealtimeModel` interface.
    """

    def __init__(
        self,
        starting_agent: RealtimeAgent,
        *,
        model: RealtimeModel | None = None,
        config: RealtimeRunConfig | None = None,
    ) -> None:
        """Initialize the realtime runner.

        Args:
            starting_agent: The agent to start the session with.
            context: The context to use for the session.
            model: The model to use. If not provided, will use a default OpenAI realtime model.
            config: Override parameters to use for the entire run.
        """
        self._starting_agent = starting_agent
        self._config = config
        self._model = model or OpenAIRealtimeWebSocketModel()

    async def run(
        self, *, context: TContext | None = None, model_config: RealtimeModelConfig | None = None
    ) -> RealtimeSession:
        """Start and returns a realtime session.

        Returns:
            RealtimeSession: A session object that allows bidirectional communication with the
            realtime model.

        Example:
            ```python
            runner = RealtimeRunner(agent)
            async with await runner.run() as session:
                await session.send_message("Hello")
                async for event in session:
                    print(event)
            ```
        """
        # Create and return the connection
        session = RealtimeSession(
            model=self._model,
            agent=self._starting_agent,
            context=context,
            model_config=model_config,
            run_config=self._config,
        )

        return session

```

--------------------------------

### Create a Realtime Agent Instance (Python)

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

Initializes a RealtimeAgent with a name and instructions. This agent will act as the AI assistant in the voice conversation. Dependencies: RealtimeAgent class.

```python
agent = RealtimeAgent(
    name="Assistant",
    instructions="You are a helpful voice assistant. Keep your responses conversational and friendly.",
)
```

--------------------------------

### RealtimeRunner Initialization

Source: https://openai.github.io/openai-agents-python/ref/realtime/runner

Initializes the RealtimeRunner, setting up the starting agent, model, and configuration for realtime agent sessions.

```APIDOC
## RealtimeRunner Constructor

### Description
Initializes the `RealtimeRunner` with a starting agent, an optional model, and configuration.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

*   **starting_agent** (`RealtimeAgent`) - Required - The agent to start the session with.
*   **model** (`RealtimeModel | None`) - Optional - The model to use. Defaults to `OpenAIRealtimeWebSocketModel` if not provided.
*   **config** (`RealtimeRunConfig | None`) - Optional - Override parameters to use for the entire run.

### Request Example
```python
from agents.realtime.runner import RealtimeRunner
from agents.realtime.agent import RealtimeAgent # Assuming RealtimeAgent is defined elsewhere

# Assuming agent is an instance of RealtimeAgent
# runner = RealtimeRunner(starting_agent=agent)
```

### Response
None (Constructor does not return a value)

```

--------------------------------

### Install OpenAI Agents SDK

Source: https://openai.github.io/openai-agents-python/index

Installs the OpenAI Agents SDK package using pip. This is the primary step to begin using the library in your Python projects.

```bash
pip install openai-agents
```

--------------------------------

### Start Trace

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/traces

Starts a trace and optionally marks it as the current trace in the execution context.

```APIDOC
## POST /trace/start

### Description
Start the trace and optionally mark it as the current trace.

### Method
POST

### Endpoint
/trace/start

### Parameters
#### Query Parameters
None

#### Request Body
- **mark_as_current** (bool) - Optional - If true, marks this trace as the current trace in the execution context. Defaults to False.

### Request Example
```json
{
  "mark_as_current": true
}
```

### Response
#### Success Response (200)
None

### Notes
- Must be called before any spans can be added
- Only one trace can be current at a time
- Thread-safe when using mark_as_current
```

--------------------------------

### Complete Example: Session Memory in Action (Python)

Source: https://openai.github.io/openai-agents-python/sessions

This example demonstrates how to use SQLiteSession for persistent conversation memory. It shows an agent remembering previous messages across multiple turns of a conversation. Requires the 'agents' library and 'asyncio'.

```python
import asyncio
from agents import Agent, Runner, SQLiteSession


async def main():
    # Create an agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )

    # Create a session instance that will persist across runs
    session = SQLiteSession("conversation_123", "conversation_history.db")

    print("=== Sessions Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    print("First turn:")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Second turn - the agent will remember the previous conversation
    print("Second turn:")
    print("User: What state is it in?")
    result = await Runner.run(
        agent,
        "What state is it in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Third turn - continuing the conversation
    print("Third turn:")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent,
        "What's the population of that state?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")


if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### Manage Turn Start (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/result

Handles the initialization logic when a new turn in the voice interaction begins. It starts tracing spans, sets internal flags, and emits a 'turn_started' event to the main queue.

```python
    async def _start_turn(self):
        if self._started_processing_turn:
            return

        self._tracing_span = speech_group_span()
        self._tracing_span.start()
        self._started_processing_turn = True
        self._first_byte_received = False
        self._generation_start_time = time_iso()
        await self._queue.put(VoiceStreamEventLifecycle(event="turn_started"))

```

--------------------------------

### Define a Basic Agent (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Defines a simple agent named 'Math Tutor' with specific instructions. Agents are the core components for performing tasks and require a name and instructions.

```python
from agents import Agent

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
```

--------------------------------

### Define Multiple Agents with Handoff Descriptions (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Defines two specialist agents, 'History Tutor' and 'Math Tutor', each with a 'handoff_description' to provide context for routing. This allows for more sophisticated agent interactions.

```python
from agents import Agent

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
```

--------------------------------

### Setup WebSocket Connection and Event Listener

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_stt

This function establishes the WebSocket connection, starts an event listener task, and waits for session creation events. It handles timeouts and other exceptions during session creation and configuration. It requires a websockets.ClientConnection object.

```python
async def _setup_connection(self, ws: websockets.ClientConnection) -> None:
        self._websocket = ws
        self._listener_task = asyncio.create_task(self._event_listener())

        try:
            event = await _wait_for_event(
                self._state_queue,
                ["session.created", "transcription_session.created"],
                SESSION_CREATION_TIMEOUT,
            )
        except TimeoutError as e:
            wrapped_err = STTWebsocketConnectionError(
                "Timeout waiting for transcription_session.created event"
            )
            await self._output_queue.put(ErrorSentinel(wrapped_err))
            raise wrapped_err from e
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise e

        await self._configure_session()

        try:
            event = await _wait_for_event(
                self._state_queue,
                ["session.updated", "transcription_session.updated"],
                SESSION_UPDATE_TIMEOUT,
            )
            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Session updated")
            else:
                logger.debug(f"Session updated: {event}")
        except TimeoutError as e:
            wrapped_err = STTWebsocketConnectionError(
                "Timeout waiting for transcription_session.updated event"
            )
            await self._output_queue.put(ErrorSentinel(wrapped_err))
            raise wrapped_err from e
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise
```

--------------------------------

### Install SQLAlchemy Sessions for OpenAI Agents

Source: https://openai.github.io/openai-agents-python/sessions/sqlalchemy_session

Installs the 'openai-agents' library with the 'sqlalchemy' extra for session storage capabilities. This is a prerequisite for using SQLAlchemy-based sessions.

```bash
pip install openai-agents[sqlalchemy]

```

--------------------------------

### Import Realtime Agent Components (Python)

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

Imports necessary classes (RealtimeAgent, RealtimeRunner) from the openai.agents.realtime module for creating and managing realtime agents. Requires Python 3.9+.

```python
import asyncio
from agents.realtime import RealtimeAgent, RealtimeRunner
```

--------------------------------

### Run Realtime Session

Source: https://openai.github.io/openai-agents-python/ref/realtime/runner

Starts and returns a RealtimeSession object, enabling bidirectional communication with the realtime model. This method is crucial for initiating and managing the interaction loop with the agent. An example demonstrates how to use the session for sending messages and receiving events.

```python
async def run(
        self, *, context: TContext | None = None, model_config: RealtimeModelConfig | None = None
    ) -> RealtimeSession:
        """Start and returns a realtime session.

        Returns:
            RealtimeSession: A session object that allows bidirectional communication with the
            realtime model.

        Example:
            ```python
            runner = RealtimeRunner(agent)
            async with await runner.run() as session:
                await session.send_message("Hello")
                async for event in session:
                    print(event)
            ```
        """
        session = RealtimeSession(
            model=self._model,
            agent=self._starting_agent,
            context=context,
            model_config=model_config,
            run_config=self._config,
        )

        return session

```

--------------------------------

### Get Conversation History Wrappers

Source: https://openai.github.io/openai-agents-python/ref/handoffs

Retrieves the current start and end markers used for the nested conversation summary.

```APIDOC
## get_conversation_history_wrappers

### Description
Return the current start/end markers used for the nested conversation summary.

### Signature
`get_conversation_history_wrappers() -> tuple[str, str]`

### Source
`src/agents/handoffs/history.py`
```

--------------------------------

### Install LiteLLM Dependency for OpenAI Agents

Source: https://openai.github.io/openai-agents-python/models/litellm

This command installs the necessary 'litellm' dependency group for the OpenAI Agents SDK, enabling the use of LiteLLmModel. This is a prerequisite for integrating external AI models.

```bash
pip install "openai-agents[litellm]"
```

--------------------------------

### Start Voice Pipeline Turn Python

Source: https://openai.github.io/openai-agents-python/ref/voice/result

Initiates a new turn in the voice pipeline. It sets up tracing, marks the turn as started, resets the first byte received flag, records the generation start time, and sends a 'turn_started' event.

```python
async def _start_turn(self):
        if self._started_processing_turn:
            return

        self._tracing_span = speech_group_span()
        self._tracing_span.start()
        self._started_processing_turn = True
        self._first_byte_received = False
        self._generation_start_time = time_iso()
        await self._queue.put(VoiceStreamEventLifecycle(event="turn_started"))
```

--------------------------------

### Custom Span Usage in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/spans

This Python example shows how to create and use a custom span for tracing operations. It utilizes a context manager (`with custom_span(...)`) to ensure the span is properly started and finished. The example illustrates setting operation-specific data during span creation and output data upon completion. It also includes a note on error handling using `set_error()`.

```python
with custom_span("database_query", {
    "operation": "SELECT",
    "table": "users"
}) as span:
    results = await db.query("SELECT * FROM users")
    span.set_output({"count": len(results)})
```

--------------------------------

### RealtimeRunner Initialization (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/runner

Initializes the RealtimeRunner, setting up the starting agent and optionally a custom model or run configuration. If no model is provided, it defaults to an OpenAIRealtimeWebSocketModel. Dependencies include RealtimeAgent, RealtimeModel, and RealtimeRunConfig.

```Python
class RealtimeRunner:
    """A `RealtimeRunner` is the equivalent of `Runner` for realtime agents. It automatically
    handles multiple turns by maintaining a persistent connection with the underlying model
    layer.

    The session manages the local history copy, executes tools, runs guardrails and facilitates
    handoffs between agents.

    Since this code runs on your server, it uses WebSockets by default. You can optionally create
    your own custom model layer by implementing the `RealtimeModel` interface.
    """

    def __init__(
        self,
        starting_agent: RealtimeAgent,
        *,
        model: RealtimeModel | None = None,
        config: RealtimeRunConfig | None = None,
    ) -> None:
        """Initialize the realtime runner.

        Args:
            starting_agent: The agent to start the session with.
            context: The context to use for the session.
            model: The model to use. If not provided, will use a default OpenAI realtime model.
            config: Override parameters to use for the entire run.
        """
        self._starting_agent = starting_agent
        self._config = config
        self._model = model or OpenAIRealtimeWebSocketModel()

```

--------------------------------

### Run Agent Orchestration (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Runs the agent orchestration defined by the 'triage_agent' to handle a user query. This demonstrates how the triage agent routes the question to the appropriate specialist agent and prints the final output.

```python
from agents import Runner

async def main():
    result = await Runner.run(triage_agent, "What is the capital of France?")
    print(result.final_output)
```

--------------------------------

### Install openai-agents with Visualization Dependencies

Source: https://openai.github.io/openai-agents-python/visualization

Installs the `openai-agents` library with the optional `viz` dependency group, which is required for agent visualization. This command uses pip, the standard Python package installer.

```bash
pip install "openai-agents[viz]"
```

--------------------------------

### RealtimeSession Example: Sending Messages and Audio, Streaming Events (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Demonstrates how to use the RealtimeSession to send messages and audio to a model and stream events back. It utilizes the RealtimeRunner to manage the session lifecycle. This example shows basic interaction patterns for realtime communication with an agent.

```python
runner = RealtimeRunner(agent)
async with await runner.run() as session:
    # Send messages
    await session.send_message("Hello")
    await session.send_audio(audio_bytes)

    # Stream events
    async for event in session:
        if event.type == "audio":
            # Handle audio event
            pass

```

--------------------------------

### Basic Trace Usage with Context Manager in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/traces

This example shows the basic usage of the `trace` context manager for defining a workflow, such as 'Order Processing'. It utilizes the `with` statement for automatic start and finish operations, ensuring reliable cleanup. The `Runner.run` method is called within the trace context to execute specific tasks.

```python
from agents.tracing import trace

# Assuming Runner, validator, and processor are defined elsewhere
# order_data = {...}

with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)

```

--------------------------------

### Generating Agent Instructions from Prompts

Source: https://openai.github.io/openai-agents-python/mcp

This example shows how to dynamically generate agent instructions using prompts provided by an MCP server. It illustrates fetching a prompt template, providing arguments to customize it, and then using the generated instructions to initialize an `Agent`. This enables flexible and context-specific agent behavior.

```python
from agents import Agent

prompt_result = await server.get_prompt(
    "generate_code_review_instructions",
    {"focus": "security vulnerabilities", "language": "python"},
)
instructions = prompt_result.messages[0].content.text

agent = Agent(
    name="Code Reviewer",
    instructions=instructions,
    mcp_servers=[server],
)

```

--------------------------------

### Initialize Agent with Hosted Tools (Python)

Source: https://openai.github.io/openai-agents-python/tools

Demonstrates how to initialize an Agent with hosted tools like WebSearchTool and FileSearchTool. These tools enable the agent to perform actions such as searching the web or retrieving information from vector stores. The example shows setting up an agent and running a task using the Runner.

```python
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
        ),
    ],
)

async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
    print(result.final_output)

```

--------------------------------

### SQLAlchemy Session Initialization from URL

Source: https://openai.github.io/openai-agents-python/sessions

Provides an example of initializing a production-ready `SQLAlchemySession` using a database connection URL. The `create_tables=True` argument ensures that the necessary database tables are created if they don't exist.

```python
from agents.extensions.memory import SQLAlchemySession

# Using database URL
session = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True
)

```

--------------------------------

### Get MCPServerStdio Name Property in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

This property returns a readable name for the MCPServerStdio instance. The name is typically derived from the command used to start the server process if not explicitly provided during initialization.

```python
@property
def name(self) -> str:
    """A readable name for the server."""
    return self._name

```

--------------------------------

### Pass OpenAI API Key Directly to Session

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

This Python code snippet shows an alternative method for authenticating with the OpenAI API by passing the API key directly when creating a session. This approach can be useful for temporary configurations or when environment variables are not accessible. It is part of the `RealtimeRunner.run()` method.

```python
session = await runner.run(model_config={"api_key": "your-api-key"})

```

--------------------------------

### Run Voice Pipeline and Stream Audio Output

Source: https://openai.github.io/openai-agents-python/voice/quickstart

Executes the voice pipeline with sample audio input (silence in this case) and streams the resulting audio output. It uses `sounddevice` to play the generated speech in real-time.

```python
import numpy as np
import sounddevice as sd
from agents.voice import AudioInput

# For simplicity, we'll just create 3 seconds of silence
# In reality, you'd get microphone data
buffer = np.zeros(24000 * 3, dtype=np.int16)
audio_input = AudioInput(buffer=buffer)

result = await pipeline.run(audio_input)

# Create an audio player using `sounddevice`
player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
player.start()

# Play the audio stream as it comes in
async for event in result.stream():
    if event.type == "voice_stream_event_audio":
        player.write(event.data)

```

--------------------------------

### Use RealtimeSession for Model Interaction (Python Example)

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/session

This example shows how to use the RealtimeSession to interact with a realtime model. It covers sending messages and audio, as well as asynchronously iterating over events streamed from the model.

```python
runner = RealtimeRunner(agent)
async with await runner.run() as session:
    # Send messages
    await session.send_message("Hello")
    await session.send_audio(audio_bytes)

    # Stream events
    async for event in session:
        if event.type == "audio":
            # Handle audio event
            pass
```

--------------------------------

### Initialize Voice Pipeline

Source: https://openai.github.io/openai-agents-python/voice/quickstart

Initializes a `VoicePipeline` using a `SingleAgentVoiceWorkflow` and the previously defined agent. This sets up the core structure for processing voice input and generating responses.

```python
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline

pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
```

--------------------------------

### Get Conversation History Wrappers in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/handoffs

Retrieves the current start and end markers used for delineating nested conversation summaries. These markers are essential for correctly parsing and reconstructing conversation history in nested scenarios.

```python
def get_conversation_history_wrappers() -> tuple[str, str]:
    """Return the current start/end markers used for the nested conversation summary."""

    return (_conversation_history_start, _conversation_history_end)
```

--------------------------------

### Get Global Trace Provider in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Retrieves the currently configured global trace provider. This function is essential for any tracing operations that rely on the global setup. It raises a RuntimeError if the trace provider has not been set.

```python
def get_trace_provider() -> TraceProvider:
    """Get the global trace provider used by tracing utilities."""
    if GLOBAL_TRACE_PROVIDER is None:
        raise RuntimeError("Trace provider not set")
    return GLOBAL_TRACE_PROVIDER
```

--------------------------------

### MCPServerStdioParams

Source: https://openai.github.io/openai-agents-python/ref/mcp/server

Defines parameters for starting an MCP server via stdio.

```APIDOC
### MCPServerStdioParams

Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another import.

#### Fields
- **command** (str) - Required - The executable to run to start the server. For example, `python` or `node`.
- **args** (list[str] | None) - Optional - Command line args to pass to the `command` executable. For example, `['foo.py']` or `['server.js', '--port', '8080']`.
- **env** (dict[str, str] | None) - Optional - The environment variables to set for the server.
- **cwd** (str | Path | None) - Optional - The working directory to use when spawning the process.
- **encoding** (str | None) - Optional - The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`.
- **encoding_error_handler** (Literal["strict", "ignore", "replace"] | None) - Optional - The text encoding error handler. Defaults to `strict`.
```

--------------------------------

### Install openai-agents with encryption extra

Source: https://openai.github.io/openai-agents-python/sessions/encrypted_session

Installs the openai-agents library with the necessary 'encrypt' extra for using encrypted sessions. This is the first step before utilizing the encryption features.

```bash
pip install openai-agents[encrypt]
```

--------------------------------

### Agent Start Event

Source: https://openai.github.io/openai-agents-python/ref/realtime/events

Details the event emitted when a new agent starts execution. Includes information about the agent and common event context.

```APIDOC
## RealtimeAgentStartEvent

### Description
A new agent has started.

### Event Type
`agent_start`

### Attributes
- **agent** (RealtimeAgent) - Required - The new agent.
- **info** (RealtimeEventInfo) - Required - Common info for all events, such as the context.

### Example
```json
{
  "agent": { ... },
  "info": { ... },
  "type": "agent_start"
}
```
```

--------------------------------

### Quick Start: Asynchronous Runner with SQLite Session

Source: https://openai.github.io/openai-agents-python/sessions

Demonstrates how to use the Runner with an SQLiteSession for multi-turn conversations. The session automatically manages history, allowing the agent to remember previous turns without manual intervention. Supports both asynchronous and synchronous runners.

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a session instance with a session ID
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Second turn - agent automatically remembers previous context
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"

# Also works with synchronous runner
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
print(result.final_output)  # "Approximately 39 million"

```

--------------------------------

### Install LiteLLM Dependency for Non-OpenAI Models

Source: https://openai.github.io/openai-agents-python/models

This command installs the necessary dependency group for integrating non-OpenAI models using LiteLLM. This allows the Agents SDK to connect with a wide range of LLM providers.

```bash
pip install "openai-agents[litellm]"

```

--------------------------------

### Start Trace

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Starts a new trace and optionally marks it as the current trace in the execution context.

```APIDOC
## POST /trace/start

### Description
Start the trace and optionally mark it as the current trace.

### Method
POST

### Endpoint
/trace/start

### Parameters
#### Query Parameters
None

#### Request Body
- **mark_as_current** (bool) - Optional - If true, marks this trace as the current trace in the execution context.

### Request Example
```json
{
  "mark_as_current": true
}
```

### Response
#### Success Response (200)
- **message** (str) - Confirmation message indicating the trace has started.

#### Response Example
{
  "message": "Trace started successfully."
}
```

--------------------------------

### Span Methods: start, finish

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

The start and finish methods control the lifecycle of a span. `start` initiates the span's execution, optionally marking it as the current span. `finish` concludes the span, optionally resetting the current span context.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """
    Start the span.

    Args:
        mark_as_current: If true, the span will be marked as the current span.
    """
    pass

@abc.abstractmethod
def finish(self, reset_current: bool = False) -> None:
    """
    Finish the span.

    Args:
        reset_current: If true, the span will be reset as the current span.
    """
    pass
```

--------------------------------

### Python: Span Base Class and Usage

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Provides the abstract base class `Span` for representing traceable operations. It outlines essential methods like `start`, `finish`, `set_error`, and context manager support (`__enter__`, `__exit__`). The example shows how to create and use custom spans for operations like database queries.

```python
class Span(abc.ABC, Generic[TSpanData]):
    """Base class for representing traceable operations with timing and context.

    A span represents a single operation within a trace (e.g., an LLM call, tool execution,
    or agent run). Spans track timing, relationships between operations, and operation-specific
    data.

    Type Args:
        TSpanData: The type of span-specific data this span contains.

    Example:
        ```python
        # Creating a custom span
        with custom_span("database_query", {
            "operation": "SELECT",
            "table": "users"
        }) as span:
            results = await db.query("SELECT * FROM users")
            span.set_output({"count": len(results)})

        # Handling errors in spans
        with custom_span("risky_operation") as span:
            try:
                result = perform_risky_operation()
            except Exception as e:
                span.set_error({
                    "message": str(e),
                    "data": {"operation": "risky_operation"}
                })
                raise
        ```

        Notes:
        - Spans automatically nest under the current trace
        - Use context managers for reliable start/finish
        - Include relevant data but avoid sensitive information
        - Handle errors properly using set_error()
    """

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """The ID of the trace this span belongs to.

        Returns:
            str: Unique identifier of the parent trace.
        """
        pass

    @property
    @abc.abstractmethod
    def span_id(self) -> str:
        """Unique identifier for this span.

        Returns:
            str: The span's unique ID within its trace.
        """
        pass

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData:
        """Operation-specific data for this span.

        Returns:
            TSpanData: Data specific to this type of span (e.g., LLM generation data).
        """
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """
        Start the span.

        Args:
            mark_as_current: If true, the span will be marked as the current span.
        """
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None:
        """
        Finish the span.

        Args:
            reset_current: If true, the span will be reset as the current span.
        """
        pass

    @abc.abstractmethod
    def __enter__(self) -> Span[TSpanData]:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    @abc.abstractmethod
    def parent_id(self) -> str | None:
        """ID of the parent span, if any.

        Returns:
            str | None: The parent span's ID, or None if this is a root span.
        """
        pass

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None:
        pass

    @property
    @abc.abstractmethod
    def error(self) -> SpanError | None:
        """Any error that occurred during span execution.

        Returns:
            SpanError | None: Error details if an error occurred, None otherwise.
        """
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        pass

    @property
    @abc.abstractmethod
    def started_at(self) -> str | None:
        """When the span started execution.

        Returns:
            str | None: ISO format timestamp of span start, None if not started.
        """
        pass

    @property
    @abc.abstractmethod
    def ended_at(self) -> str | None:
        """When the span finished execution.

        Returns:
            str | None: ISO format timestamp of span end, None if not finished.
        """
        pass
```

--------------------------------

### GET /list_tools

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Retrieves a list of available tools on the server.

```APIDOC
## GET /list_tools

### Description
Lists the tools available on the server. This can be filtered based on the provided context and agent.

### Method
GET

### Endpoint
`/list_tools`

### Parameters
#### Query Parameters
- **run_context** (RunContextWrapper[Any] | None) - Optional - The run context wrapper.
- **agent** (AgentBase | None) - Optional - The agent instance.

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available tools.

#### Response Example
```json
{
  "tools": [
    {
      "name": "example_tool",
      "description": "An example tool.",
      "parameters": {
        "type": "object",
        "properties": {
          "input": {"type": "string"}
        },
        "required": ["input"]
      }
    }
  ]
}
```

#### Error Response (400)
- **error** (string) - Description of the error if the server is not initialized.

#### Error Response Example
```json
{
  "error": "Server not initialized. Make sure you call `connect()` first."
}
```
```

--------------------------------

### Start Trace - Python

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/traces

Starts a trace and optionally marks it as the current trace in the execution context. This method must be called before adding any spans, and it ensures proper context management, allowing only one trace to be current at a time.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """Start the trace and optionally mark it as the current trace.

    Args:
        mark_as_current: If true, marks this trace as the current trace
            in the execution context.

    Notes:
        - Must be called before any spans can be added
        - Only one trace can be current at a time
        - Thread-safe when using mark_as_current
    """
    pass
```

--------------------------------

### Branch Workflow Example: Create, Switch, Continue

Source: https://openai.github.io/openai-agents-python/sessions/advanced_sqlite_session

Illustrates a typical branch workflow: initiating a conversation, storing run usage, creating a new branch from a specific turn, continuing the conversation in the new branch, and switching back to the main branch. Requires `Runner` and `session` objects.

```python
# Original conversation
result = await Runner.run(agent, "What's the capital of France?", session=session)
await session.store_run_usage(result)

result = await Runner.run(agent, "What's the weather like there?", session=session)
await session.store_run_usage(result)

# Create branch from turn 2 (weather question)
branch_id = await session.create_branch_from_turn(2, "weather_focus")

# Continue in new branch with different question
result = await Runner.run(
    agent, 
    "What are the main tourist attractions in Paris?", 
    session=session
)
await session.store_run_usage(result)

# Switch back to main branch
await session.switch_to_branch("main")

# Continue original conversation
result = await Runner.run(
    agent, 
    "How expensive is it to visit?", 
    session=session
)
await session.store_run_usage(result)
```

--------------------------------

### Set OpenAI API Key using Environment Variable

Source: https://openai.github.io/openai-agents-python/realtime/quickstart

This command demonstrates how to set the OpenAI API key using an environment variable. This is a common practice for securely managing API keys, especially in development and production environments. The `export` command is used in Unix-like shells (Linux, macOS) to set the variable for the current session.

```bash
export OPENAI_API_KEY="your-api-key-here"

```

--------------------------------

### Start Trace Method (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/traces

Abstract method to start a trace. It can optionally mark the trace as the current one in the execution context. This method must be called before adding any spans and ensures thread-safety when `mark_as_current` is used.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """Start the trace and optionally mark it as the current trace.

    Args:
        mark_as_current: If true, marks this trace as the current trace
            in the execution context.

    Notes:
        - Must be called before any spans can be added
        - Only one trace can be current at a time
        - Thread-safe when using mark_as_current
    """
    pass

```

--------------------------------

### Use Non-OpenAI Models with LiteLLM Prefix

Source: https://openai.github.io/openai-agents-python/models

This example shows how to instantiate Agents with non-OpenAI models by prefixing the model name with 'litellm/'. It demonstrates using Anthropic's Claude and Google's Gemini models.

```python
claude_agent = Agent(model="litellm/anthropic/claude-3-5-sonnet-20240620", ...)
gemini_agent = Agent(model="litellm/gemini/gemini-2.5-flash-preview-04-17", ...)

```

--------------------------------

### Quick start: Encrypted session with SQLAlchemy

Source: https://openai.github.io/openai-agents-python/sessions/encrypted_session

Demonstrates how to set up and use an EncryptedSession with a SQLAlchemy session backend. It shows the creation of an underlying session, wrapping it with EncryptedSession, and then using it with the Runner.

```python
import asyncio
from agents import Agent, Runner
from agents.extensions.memory import EncryptedSession, SQLAlchemySession

async def main():
    agent = Agent("Assistant")

    # Create underlying session
    underlying_session = SQLAlchemySession.from_url(
        "user-123",
        url="sqlite+aiosqlite:///:memory:",
        create_tables=True
    )

    # Wrap with encryption
    session = EncryptedSession(
        session_id="user-123",
        underlying_session=underlying_session,
        encryption_key="your-secret-key-here",
        ttl=600  # 10 minutes
    )

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

--------------------------------

### GET /tools/all

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/agent

Retrieves all agent tools, including both MCP tools and function tools. This endpoint provides a consolidated list of all tools that the agent can utilize.

```APIDOC
## GET /tools/all

### Description
All agent tools, including MCP tools and function tools.

### Method
GET

### Endpoint
/tools/all

### Parameters
#### Query Parameters
- **run_context** (RunContextWrapper) - Required - The context for the current run.

### Request Example
```json
{
  "run_context": { "type": "RunContextWrapper" } 
}
```

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of all available agent tools, including MCP and function tools.

#### Response Example
```json
[
  { "type": "Tool", "name": "example_mcp_tool", "description": "An example MCP tool" },
  { "type": "FunctionTool", "name": "example_function_tool", "description": "An example function tool", "is_enabled": true },
  { "type": "FunctionTool", "name": "another_function_tool", "description": "Another function tool", "is_enabled": false }
]
```
```

--------------------------------

### Define Agents and Tools for Voice Interaction

Source: https://openai.github.io/openai-agents-python/voice/quickstart

Sets up custom agents, including a Spanish-speaking agent and a primary assistant agent, with a weather-fetching tool. The assistant agent is configured to handoff to the Spanish agent if the user speaks Spanish. This defines the conversational logic for the voice pipeline.

```python
import asyncio
import random

from agents import (
    Agent,
    function_tool,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions



@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-5.2",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-5.2",
    handoffs=[spanish_agent],
    tools=[get_weather],
)

```

--------------------------------

### Implement Guardrail Function (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Implements a guardrail function 'homework_guardrail' that uses the 'guardrail_agent' to check user input. It returns a 'GuardrailFunctionOutput' indicating if the guardrail was triggered.

```python
from agents import GuardrailFunctionOutput, Agent, Runner
from pydantic import BaseModel


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )
```

--------------------------------

### Get AsyncOpenAI Client (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/models/openai_chatcompletions

This helper function ensures that an `AsyncOpenAI` client instance is created and returned. If the client has not been initialized, it creates a new one; otherwise, it returns the existing instance.

```python
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

```

--------------------------------

### Trace Class Example Usage (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Example demonstrating how to use the Trace class to manage a sequence of operations within a logical workflow.

```python
# Basic trace usage
with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)

```

--------------------------------

### OpenAI Provider Initialization (__init__)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/multi_provider

Initializes a new OpenAI provider with various configuration options for API keys, base URLs, clients, organizations, projects, and response usage.

```APIDOC
## POST /openai/provider/initialize

### Description
Initializes a new OpenAI provider. You can configure API keys, base URLs, and other settings.

### Method
POST

### Endpoint
/openai/provider/initialize

### Parameters
#### Request Body
- **provider_map** (MultiProviderMap | None) - Optional - A MultiProviderMap that maps prefixes to ModelProviders. If not provided, a default mapping will be used.
- **openai_api_key** (str | None) - Optional - The API key to use for the OpenAI provider. If not provided, the default API key will be used.
- **openai_base_url** (str | None) - Optional - The base URL to use for the OpenAI provider. If not provided, the default base URL will be used.
- **openai_client** (AsyncOpenAI | None) - Optional - An optional OpenAI client to use. If not provided, a new OpenAI client will be created using the api_key and base_url.
- **openai_organization** (str | None) - Optional - The organization to use for the OpenAI provider.
- **openai_project** (str | None) - Optional - The project to use for the OpenAI provider.
- **openai_use_responses** (bool | None) - Optional - Whether to use the OpenAI responses API.

### Request Example
```json
{
  "openai_api_key": "your_api_key",
  "openai_base_url": "https://api.openai.com/v1",
  "openai_organization": "your_organization_id"
}
```

### Response
#### Success Response (200)
- **message** (str) - Confirmation message indicating successful initialization.

#### Response Example
```json
{
  "message": "OpenAI provider initialized successfully."
}
```
```

--------------------------------

### Response Span Creation

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

This endpoint allows for the creation of a new response span. The span is not automatically started and requires manual initiation using a `with` statement or by calling `start()` and `finish()` methods.

```APIDOC
## response_span

### Description
Create a new response span. The span will not be started automatically, you should either do `with response_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
N/A (Function Call)

### Endpoint
N/A (Function Call)

#### Parameters

##### Arguments
- **response** (`Response | None`) - Optional - The OpenAI Response object.
- **span_id** (`str | None`) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (`Trace | Span[Any] | None`) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (`bool`) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to `False`.

### Request Example
```python
# Example usage within a `with` statement
with response_span(response=my_response) as span:
    # Span is automatically started and finished
    pass

# Example usage with manual start/finish
span = response_span(response=my_response)
span.start()
# ... do some work ...
span.finish()
```

### Response
#### Success Response (Span Object)
- **Span[ResponseSpanData]** - The newly created response span object.
```

--------------------------------

### RealtimeRunner Constructor

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/runner

The constructor for RealtimeRunner initializes the runner with a starting agent and optional model and configuration parameters. It defaults to using an OpenAI realtime WebSocket model if no model is provided.

```python
def __init__(
    self,
    starting_agent: RealtimeAgent,
    *,
    model: RealtimeModel | None = None,
    config: RealtimeRunConfig | None = None,
) -> None:
    """Initialize the realtime runner.

    Args:
        starting_agent: The agent to start the session with.
        context: The context to use for the session.
        model: The model to use. If not provided, will use a default OpenAI realtime model.
        config: Override parameters to use for the entire run.
    """
    self._starting_agent = starting_agent
    self._config = config
    self._model = model or OpenAIRealtimeWebSocketModel()

```

--------------------------------

### Get Prompt

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Fetches a specific prompt from the server by its name, optionally with arguments. Requires an initialized server session.

```APIDOC
## GET /api/prompts/{name}

### Description
Get a specific prompt from the server.

### Method
GET

### Endpoint
/api/prompts/{name}

### Parameters
#### Path Parameters
- **name** (str) - Required - The name of the prompt to retrieve.
#### Query Parameters
- **arguments** (dict[str, Any] | None) - Optional - A dictionary of arguments to pass to the prompt.

### Response
#### Success Response (200)
- **GetPromptResult** (object) - An object containing the details of the requested prompt.
```

--------------------------------

### Connect to Server Asynchronously - Python

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Establishes an asynchronous connection to the server. It utilizes the `create_streams` method to get communication streams and initializes a `ClientSession`. Error handling is included to log issues during initialization and trigger cleanup if necessary.

```python
async def connect(self):
    """Connect to the server."""
    try:
        transport = await self.exit_stack.enter_async_context(self.create_streams())
        # streamablehttp_client returns (read, write, get_session_id)
        # sse_client returns (read, write)

        read, write, *_ = transport

        session = await self.exit_stack.enter_async_context(
            ClientSession(
                read,
                write,
                timedelta(seconds=self.client_session_timeout_seconds)
                if self.client_session_timeout_seconds
                else None,
                message_handler=self.message_handler,
            )
        )
        server_result = await session.initialize()
        self.server_initialize_result = server_result
        self.session = session
    except Exception as e:
        logger.error(f"Error initializing MCP server: {e}")
        await self.cleanup()
        raise

```

--------------------------------

### Initialize OpenAIVoiceModelProvider in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_model_provider

Demonstrates the initialization of the OpenAIVoiceModelProvider class. It supports direct instantiation with an existing OpenAI client or configuration via API key, base URL, organization, and project. It includes a check to prevent providing both an API key/base URL and an existing client.

```python
class OpenAIVoiceModelProvider(VoiceModelProvider):
    """A voice model provider that uses OpenAI models."""

    def __init__(
        self,
        *, 
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        """Create a new OpenAI voice model provider.

        Args:
            api_key: The API key to use for the OpenAI client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
                default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            organization: The organization to use for the OpenAI client.
            project: The project to use for the OpenAI client.
        """
        if openai_client is not None:
            assert api_key is None and base_url is None, (
                "Don't provide api_key or base_url if you provide openai_client"
            )
            self._client: AsyncOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project
```

--------------------------------

### Get MCP Tools

Source: https://openai.github.io/openai-agents-python/ja/ref/agent

Fetches the available tools from the MCP servers.

```APIDOC
## GET /tools/mcp

### Description
Fetches the available tools from the MCP servers.

### Method
GET

### Endpoint
/tools/mcp

### Parameters
#### Query Parameters
- **run_context** (RunContextWrapper) - Required - The run context for the request.

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available MCP tools.

#### Response Example
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "example_tool",
        "description": "An example tool.",
        "parameters": {
          "type": "object",
          "properties": {
            "arg1": {
              "type": "string",
              "description": "An example argument."
            }
          },
          "required": ["arg1"]
        }
      }
    }
  ]
}
```
```

--------------------------------

### GET /tools/mcp

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/agent

Fetches the available tools from the MCP servers. This endpoint retrieves a list of tools that are accessible via the MCP (Multi-Cloud Platform) infrastructure.

```APIDOC
## GET /tools/mcp

### Description
Fetches the available tools from the MCP servers.

### Method
GET

### Endpoint
/tools/mcp

### Parameters
#### Query Parameters
- **run_context** (RunContextWrapper) - Required - The context for the current run.

### Request Example
```json
{
  "run_context": { "type": "RunContextWrapper" } 
}
```

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available tools from MCP servers.

#### Response Example
```json
[
  { "type": "Tool", "name": "example_mcp_tool", "description": "An example MCP tool" },
  { "type": "Tool", "name": "another_mcp_tool", "description": "Another MCP tool" }
]
```
```

--------------------------------

### Get Response from Model

Source: https://openai.github.io/openai-agents-python/ja/ref/models/interface

This endpoint allows you to get a response from the model by providing system instructions, input, model settings, tools, and other configuration parameters.

```APIDOC
## POST /api/agents/response

### Description
Retrieves a response from the language model based on the provided configuration and input.

### Method
POST

### Endpoint
/api/agents/response

### Parameters
#### Request Body
- **system_instructions** (str | None) - Required - The system instructions to guide the model's behavior.
- **input** (str | list[TResponseInputItem]) - Required - The input data for the model, formatted as specified.
- **model_settings** (ModelSettings) - Required - Configuration settings for the language model.
- **tools** (list[Tool]) - Required - A list of available tools the model can utilize.
- **output_schema** (AgentOutputSchemaBase | None) - Required - Defines the expected schema for the model's output.
- **handoffs** (list[Handoff]) - Required - A list of handoff configurations for managing conversation flow.
- **tracing** (ModelTracing) - Required - Configuration for tracing model interactions.
- **previous_response_id** (str | None) - Required - The ID of the prior response, used in specific scenarios like the OpenAI Responses API.
- **conversation_id** (str | None) - Required - The identifier for an ongoing conversation, if applicable.
- **prompt** (ResponsePromptParam | None) - Required - Custom prompt configuration to influence the model's response.

### Request Example
```json
{
  "system_instructions": "You are a helpful assistant.",
  "input": "What is the capital of France?",
  "model_settings": { "temperature": 0.7 },
  "tools": [],
  "output_schema": null,
  "handoffs": [],
  "tracing": { "enabled": true },
  "previous_response_id": null,
  "conversation_id": "conv_123",
  "prompt": null
}
```

### Response
#### Success Response (200)
- **response** (ModelResponse) - The complete response generated by the model.

#### Response Example
```json
{
  "response": {
    "output": {
      "text": "The capital of France is Paris."
    },
    "metadata": {}
  }
}
```
```

--------------------------------

### Agent with Dynamic Instructions

Source: https://openai.github.io/openai-agents-python/agents

Demonstrates how to provide dynamic instructions to an agent using a function. The example defines a `dynamic_instructions` function that generates instructions based on the provided context and agent, allowing for more flexible and context-aware agent behavior.

```python
from agents import Agent, RunContextWrapper
from dataclasses import dataclass

@dataclass
class UserContext:
    name: str
    uid: str
    is_pro_user: bool

def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)

```

--------------------------------

### Connect OpenAIRealtimeWebSocketModel

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/openai_realtime

Establishes a WebSocket connection to the OpenAI API using the provided configuration options. This method handles initial setup, including setting the model name and potentially attaching to an existing real-time call, ensuring the connection is not already established.

```python
    async def connect(self, options: RealtimeModelConfig) -> None:
        """Establish a connection to the model and keep it alive."""
        assert self._websocket is None, "Already connected"
        assert self._websocket_task is None, "Already connected"

        model_settings: RealtimeSessionModelSettings = options.get("initial_model_settings", {})

        self._playback_tracker = options.get("playback_tracker", None)

        call_id = options.get("call_id")
        model_name = model_settings.get("model_name")
        if call_id and model_name:
            error_message = (
                "Cannot specify both `call_id` and `model_name` "
                "when attaching to an existing realtime call."
            )
            raise UserError(error_message)

        if model_name:
            self.model = model_name

        self._call_id = call_id
```

--------------------------------

### Create Custom Span

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

The `custom_span` function allows you to create a new span with custom metadata. This span is not automatically started and requires manual starting and finishing or usage within a `with` statement.

```APIDOC
## POST /custom_span

### Description
Creates a new custom span with associated metadata. The span requires manual start/finish operations or context management.

### Method
POST

### Endpoint
/custom_span

#### Parameters

#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **name** (str) - Required - The name of the custom span.
- **data** (dict[str, Any] | None) - Optional - Arbitrary structured data to associate with the span. Defaults to None.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated. Defaults to None.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span is used. Defaults to None.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "name": "my_custom_operation",
  "data": {
    "key1": "value1",
    "key2": 123
  },
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span** (Span[CustomSpanData]) - The newly created custom span.

#### Response Example
```json
{
  "span": {
    "name": "my_custom_operation",
    "data": {
      "key1": "value1",
      "key2": 123
    },
    "id": "generated_span_id"
  }
}
```
```

--------------------------------

### GET /api/history

Source: https://openai.github.io/openai-agents-python/ref/memory

Retrieves the conversation history for the current session. You can limit the number of items returned to get the latest entries.

```APIDOC
## GET /api/history

### Description
Retrieve the conversation history for this session. You can optionally specify a limit to retrieve only the most recent items.

### Method
GET

### Endpoint
/api/history

### Parameters
#### Query Parameters
- **limit** (integer) - Optional - Maximum number of items to retrieve. If not specified, retrieves all items. When specified, returns the latest N items in chronological order.

### Request Example
```
GET /api/history?limit=10
```

### Response
#### Success Response (200)
- **items** (list[object]) - List of input items representing the conversation history. Each item is an object with conversation data.

#### Response Example
```json
{
  "items": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking!"
    }
  ]
}
```
```

--------------------------------

### POST /connect

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Connects to the server and initializes the client session.

```APIDOC
## POST /connect

### Description
Connects to the server and establishes a client session. This method should be called before other API operations.

### Method
POST

### Endpoint
`/connect`

### Parameters
*This endpoint does not take explicit parameters, but relies on the server being initialized via `create_streams` and available context.* 

### Response
#### Success Response (200)
*Indicates successful connection and session initialization.*

#### Response Example
*No specific JSON response body is defined for success, the operation's success is indicated by the absence of an error.*
```

--------------------------------

### Initialize Tracing Processor with Parameters

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processors

Initializes a tracing processor, setting up parameters for queue size, batch size, schedule delay, and export trigger ratio. It configures the internal queue and determines the threshold for triggering exports based on the queue size.

```python
def __init__(
    self,
    exporter: TracingExporter,
    max_queue_size: int = 8192,
    max_batch_size: int = 128,
    schedule_delay: float = 5.0,
    export_trigger_ratio: float = 0.7,
):
    """
    Args:
        exporter: The exporter to use.
        max_queue_size: The maximum number of spans to store in the queue. After this, we will
            start dropping spans.
        max_batch_size: The maximum number of spans to export in a single batch.
        schedule_delay: The delay between checks for new spans to export.
        export_trigger_ratio: The ratio of the queue size at which we will trigger an export.
    """
    self._exporter = exporter
    self._queue: queue.Queue[Trace | Span[Any]] = queue.Queue(maxsize=max_queue_size)
    self._max_queue_size = max_queue_size
    self._max_batch_size = max_batch_size
    self._schedule_delay = schedule_delay
    self._shutdown_event = threading.Event()

    # The queue size threshold at which we export immediately.
    self._export_trigger_size = max(1, int(max_queue_size * export_trigger_ratio))

    # Track when we next *must* perform a scheduled export
    self._next_export_time = time.time() + self._schedule_delay

    # We lazily start the background worker thread the first time a span/trace is queued.
    self._worker_thread: threading.Thread | None = None
    self._thread_start_lock = threading.Lock()

```

--------------------------------

### Agent Span Creation API

Source: https://openai.github.io/openai-agents-python/ref/tracing

This API endpoint allows for the creation of a new agent span. The span is not started automatically and requires manual management via `with agent_span(...)` or explicit `start()` and `finish()` calls.

```APIDOC
## POST /v1/tracing/agent_span

### Description
Creates a new agent span for tracing purposes. This span needs to be manually started and finished.

### Method
POST

### Endpoint
/v1/tracing/agent_span

### Parameters
#### Request Body
- **name** (str) - Required - The name of the agent.
- **handoffs** (list[str] | None) - Optional - A list of agent names to which this agent could hand off control.
- **tools** (list[str] | None) - Optional - A list of tool names available to this agent.
- **output_type** (str | None) - Optional - The name of the output type produced by the agent.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. Defaults to the current trace/span.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "name": "MyAgent",
  "handoffs": ["AnotherAgent"],
  "tools": ["Tool1", "Tool2"],
  "output_type": "string",
  "span_id": "generated-span-id-123",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[AgentSpanData]** - The newly created agent span object.

#### Response Example
```json
{
  "span_id": "generated-span-id-123",
  "name": "MyAgent",
  "start_time": "2023-10-27T10:00:00Z",
  "end_time": null,
  "status": "running"
}
```
```

--------------------------------

### RealtimeRunner.run()

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/runner

Starts and returns a realtime session, enabling bidirectional communication with the realtime model. This method is asynchronous and should be used within an async context.

```APIDOC
## RealtimeRunner.run()

### Description
Starts and returns a realtime session. This session object allows for bidirectional communication with the realtime model.

### Method
`async def run(
    self,
    *,
    context: TContext | None = None,
    model_config: RealtimeModelConfig | None = None,
) -> RealtimeSession`

### Endpoint
N/A (This is a Python method, not a REST endpoint)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
runner = RealtimeRunner(agent)
async with await runner.run() as session:
    await session.send_message("Hello")
    async for event in session:
        print(event)
```

### Response
#### Success Response (RealtimeSession)
- **RealtimeSession** (`RealtimeSession`) - A session object that allows bidirectional communication with the realtime model.

#### Response Example
```json
// The actual response is a RealtimeSession object, not a JSON payload.
// See the Request Example for usage.
```
```

--------------------------------

### RealtimeSession Async Context Manager Entry

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Asynchronously enters the session context, connecting to the model and enabling event streaming and message/audio exchange. This is the recommended way to start a session.

```APIDOC
## RealtimeSession `__aenter__` (async)

### Description
Start the session by connecting to the model. After this, you will be able to stream events from the model and send messages and audio to the model. This method is intended for use with the `async with` statement.

### Method
`__aenter__`

### Endpoint
N/A (Instance method)

### Parameters
None

### Request Example
None

### Response
#### Success Response (200)
- **self** (`RealtimeSession`) - The current session instance.
```

--------------------------------

### Create Transcription Span (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

Creates a new span for speech-to-text transcription. The span requires manual starting and finishing using `with` statements or `start()`/`finish()` calls. It accepts model details, audio input, input format, output, model configuration, span ID, and parent trace/span as parameters.

```python
from typing import Any, Mapping
from opentelemetry.trace import Trace, Span

# Assuming necessary imports for Span, Trace, TranscriptionSpanData, and get_trace_provider
# from .span import Span
# from .trace import Trace
# from .data import TranscriptionSpanData
# from .provider import get_trace_provider


def transcription_span(
    model: str | None = None,
    input: str | None = None,
    input_format: str | None = "pcm",
    output: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[TranscriptionSpanData]:
    """Create a new transcription span. The span will not be started automatically, you should
    either do `with transcription_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        model: The name of the model used for the speech-to-text.
        input: The audio input of the speech-to-text transcription, as a base64 encoded string of
            audio bytes.
        input_format: The format of the audio input (defaults to "pcm").
        output: The output of the speech-to-text transcription.
        model_config: The model configuration (hyperparameters) used.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created speech-to-text span.
    """
    # Placeholder for actual implementation, assuming get_trace_provider() and TranscriptionSpanData exist
    # return get_trace_provider().create_span(
    #     span_data=TranscriptionSpanData(
    #         input=input,
    #         input_format=input_format,
    #         output=output,
    #         model=model,
    #         model_config=model_config,
    #     ),
    #     span_id=span_id,
    #     parent=parent,
    #     disabled=disabled,
    # )
    pass # Replace with actual implementation

```

--------------------------------

### Start Transcription Turn in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/voice/models/openai_stt

Initiates a new turn in the transcription process by creating and starting a tracing span. This span captures model configuration details like temperature, language, prompt, and turn detection settings.

```python
def _start_turn(self) -> None:
        self._tracing_span = transcription_span(
            model=self._model,
            model_config={
                "temperature": self._settings.temperature,
                "language": self._settings.language,
                "prompt": self._settings.prompt,
                "turn_detection": self._turn_detection,
            },
        )
        self._tracing_span.start()

```

--------------------------------

### SQLAlchemySession Initialization

Source: https://openai.github.io/openai-agents-python/zh/ref/extensions/memory/sqlalchemy_session

Creates an instance of SQLAlchemySession, establishing a connection to the database using SQLAlchemy's async engine. It handles optional keyword arguments for engine creation and constructor parameters. Dependencies include `sqlalchemy.ext.asyncio.create_async_engine`.

```python
def __init__(
        cls,
        session_id: str,
        url: str,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "SQLAlchemySession":
        """Create a new SQLAlchemySession.

        Args:
            session_id: The unique identifier for the session.
            url: The database connection URL.
            engine_kwargs (dict[str, Any] | None): Additional keyword arguments forwarded to
                sqlalchemy.ext.asyncio.create_async_engine.
            **kwargs: Additional keyword arguments forwarded to the main constructor
                (e.g., create_tables, custom table names, etc.).

        Returns:
            SQLAlchemySession: An instance of SQLAlchemySession connected to the specified database.
        """
        engine_kwargs = engine_kwargs or {}
        engine = create_async_engine(url, **engine_kwargs)
        return cls(session_id, engine=engine, **kwargs)
```

--------------------------------

### Get Prompt API

Source: https://openai.github.io/openai-agents-python/ref/mcp/server

Retrieves a specific prompt from the server by name.

```APIDOC
## GET /prompts/{name}

### Description
Get a specific prompt from the server.

### Method
GET

### Endpoint
/prompts/{name}

### Parameters
#### Path Parameters
- **name** (str) - Required - The name of the prompt to retrieve.
#### Query Parameters
- **arguments** (dict[str, Any] | None) - Optional - A dictionary of arguments to use when retrieving the prompt.

### Response
#### Success Response (200)
- **prompt_details** (GetPromptResult) - The details of the requested prompt.
```

--------------------------------

### Basic Trace Usage in Python

Source: https://openai.github.io/openai-agents-python/ref/tracing/traces

Demonstrates the fundamental usage of the `trace` context manager to define and execute a logical workflow. It shows how to start a trace with a descriptive name and execute asynchronous operations within its scope.

```python
from opentelemetry.trace import trace
from agents.runner import Runner

# Basic trace usage
with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)

```

--------------------------------

### Start Transcription Turn with Tracing (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_stt

Initiates a new transcription turn by creating a tracing span. This span records model details, configuration settings like temperature, language, prompt, and turn detection. The span is started immediately to capture the beginning of the transcription process. Dependencies include tracing utilities.

```python
def _start_turn(self) -> None:
        self._tracing_span = transcription_span(
            model=self._model,
            model_config={
                "temperature": self._settings.temperature,
                "language": self._settings.language,
                "prompt": self._settings.prompt,
                "turn_detection": self._turn_detection,
            },
        )
        self._tracing_span.start()
```

--------------------------------

### SQLiteSession Initialization and Database Setup (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/memory/sqlite_session

Initializes the SQLiteSession with session details and configures the SQLite database. It handles both in-memory databases (losing data on process end) and persistent file-based databases. The schema for sessions and messages tables is created if they don't exist, with journal_mode set to WAL for performance.

```python
class SQLiteSession(SessionABC):
    """SQLite-based implementation of session storage.

    This implementation stores conversation history in a SQLite database.
    By default, uses an in-memory database that is lost when the process ends.
    For persistent storage, provide a file path.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str | Path = ":memory:",
        sessions_table: str = "agent_sessions",
        messages_table: str = "agent_messages",
    ):
        """Initialize the SQLite session.

        Args:
            session_id: Unique identifier for the conversation session
            db_path: Path to the SQLite database file. Defaults to ':memory:' (in-memory database)
            sessions_table: Name of the table to store session metadata. Defaults to
                'agent_sessions'
            messages_table: Name of the table to store message data. Defaults to 'agent_messages'
        """
        self.session_id = session_id
        self.db_path = db_path
        self.sessions_table = sessions_table
        self.messages_table = messages_table
        self._local = threading.local()
        self._lock = threading.Lock()

        # For in-memory databases, we need a shared connection to avoid thread isolation
        # For file databases, we use thread-local connections for better concurrency
        self._is_memory_db = str(db_path) == ":memory:"
        if self._is_memory_db:
            self._shared_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_connection.execute("PRAGMA journal_mode=WAL")
            self._init_db_for_connection(self._shared_connection)
        else:
            # For file databases, initialize the schema once since it persists
            init_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            init_conn.execute("PRAGMA journal_mode=WAL")
            self._init_db_for_connection(init_conn)
            init_conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._is_memory_db:
            # Use shared connection for in-memory database to avoid thread isolation
            return self._shared_connection
        else:
            # Use thread-local connections for file databases
            if not hasattr(self._local, "connection"):
                self._local.connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                )
                self._local.connection.execute("PRAGMA journal_mode=WAL")
            assert isinstance(self._local.connection, sqlite3.Connection), (
                f"Expected sqlite3.Connection, got {type(self._local.connection)}"
            )
            return self._local.connection

    def _init_db_for_connection(self, conn: sqlite3.Connection) -> None:
        """Initialize the database schema for a specific connection."""
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

```

--------------------------------

### Analyze Conversation: Get by Turns, Tool Usage, Find by Content

Source: https://openai.github.io/openai-agents-python/sessions/advanced_sqlite_session

Shows how to perform structured queries on conversation data using `AdvancedSQLiteSession`. Methods include retrieving conversation organized by turns, getting tool usage statistics, and finding turns by matching content. Depends on the `session` object.

```python
# Get conversation organized by turns
conversation_by_turns = await session.get_conversation_by_turns()
for turn_num, items in conversation_by_turns.items():
    print(f"Turn {turn_num}: {len(items)} items")
    for item in items:
        if item["tool_name"]:
            print(f"  - {item['type']} (tool: {item['tool_name']})")
        else:
            print(f"  - {item['type']}")

# Get tool usage statistics
tool_usage = await session.get_tool_usage()
for tool_name, count, turn in tool_usage:
    print(f"{tool_name}: used {count} times in turn {turn}")

# Find turns by content
matching_turns = await session.find_turns_by_content("weather")
for turn in matching_turns:
    print(f"Turn {turn['turn']}: {turn['content']}")
```

--------------------------------

### Initialize OpenAI Voice Model Provider (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_provider

Initializes the OpenAI voice model provider. It accepts an API key, base URL, an optional pre-configured OpenAI client, organization, and project. If an `openai_client` is provided, API key and base URL should not be specified.

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    organization: str | None = None,
    project: str | None = None,
) -> None:
    """Create a new OpenAI voice model provider.

    Args:
        api_key: The API key to use for the OpenAI client. If not provided, we will use the
            default API key.
        base_url: The base URL to use for the OpenAI client. If not provided, we will use the
            default base URL.
        openai_client: An optional OpenAI client to use. If not provided, we will create a new
            OpenAI client using the api_key and base_url.
        organization: The organization to use for the OpenAI client.
        project: The project to use for the OpenAI client.
    """
    if openai_client is not None:
        assert api_key is None and base_url is None,
            "Don't provide api_key or base_url if you provide openai_client"
        self._client: AsyncOpenAI | None = openai_client
    else:
        self._client = None
        self._stored_api_key = api_key
        self._stored_base_url = base_url
        self._stored_organization = organization
        self._stored_project = project

```

--------------------------------

### Get Function Tools

Source: https://openai.github.io/openai-agents-python/ref/mcp/util

Retrieves all function tools available from a single MCP server.

```APIDOC
## GET /tools

### Description
Get all function tools from a single MCP server.

### Method
GET

### Endpoint
/tools

### Parameters
#### Query Parameters
- **server** (MCPServer) - Required - The MCP server object to retrieve tools from.
- **convert_schemas_to_strict** (bool) - Required - Flag to determine if input schemas should be converted to a strict JSON schema.
- **run_context** (RunContextWrapper[Any]) - Required - The runtime context for the operation.
- **agent** (AgentBase) - Required - The agent instance performing the operation.

### Request Example
```json
{
  "server": {
    "name": "server1",
    "list_tools": "async function() { ... }"
  },
  "convert_schemas_to_strict": false,
  "run_context": "...",
  "agent": "..."
}
```

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of Tool objects representing the function tools found on the server.

#### Response Example
```json
[
  {
    "name": "tool1",
    "description": "Description of tool1",
    "params_json_schema": { ... },
    "on_invoke_tool": "function() { ... }",
    "strict_json_schema": false
  }
]
```
```

--------------------------------

### RealtimeRunner Initialization Method (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/runner

This Python code snippet details the `__init__` method for the `RealtimeRunner` class. It outlines the parameters required for initializing a real-time agent session, including the starting agent, an optional custom model, and configuration overrides. The method initializes internal state and sets up a default OpenAI realtime WebSocket model if none is provided. The arguments specify the type and purpose of each parameter.

```python
def __init__(
    self,
    starting_agent: RealtimeAgent,
    *,
    model: RealtimeModel | None = None,
    config: RealtimeRunConfig | None = None,
) -> None:
    """Initialize the realtime runner.

    Args:
        starting_agent: The agent to start the session with.
        context: The context to use for the session.
        model: The model to use. If not provided, will use a default OpenAI realtime model.
        config: Override parameters to use for the entire run.
    """
    self._starting_agent = starting_agent
    self._config = config
    self._model = model or OpenAIRealtimeWebSocketModel()

```

--------------------------------

### Basic Trace Workflow Example in Python

Source: https://openai.github.io/openai-agents-python/ref/tracing

Illustrates the fundamental usage of the `trace` context manager for basic workflow tracking. It shows how to provide a workflow name and execute asynchronous operations within the traced block. This pattern is useful for monitoring the execution of distinct parts of an application.

```python
from openai_agents.tracing import trace

# ... other imports and setup ...

with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)
```

--------------------------------

### Creating Handoffs Between Realtime Agents (Python)

Source: https://openai.github.io/openai-agents-python/realtime/guide

Illustrates how to implement conversation handoffs between specialized `RealtimeAgent` instances. This allows a main agent to transfer the conversation to a more suitable agent (e.g., billing or technical support) based on user intent. The example defines specialized agents and then configures the main agent with handoff configurations.

```python
from agents.realtime import RealtimeAgent, realtime_handoff

# Specialized agents
billing_agent = RealtimeAgent(
    name="Billing Support",
    instructions="You specialize in billing and payment issues.",
)

technical_agent = RealtimeAgent(
    name="Technical Support",
    instructions="You handle technical troubleshooting.",
)

# Main agent with handoffs
main_agent = RealtimeAgent(
    name="Customer Service",
    instructions="You are the main customer service agent. Hand off to specialists when needed.",
    handoffs=[
        realtime_handoff(billing_agent, tool_description="Transfer to billing support"),
        realtime_handoff(technical_agent, tool_description="Transfer to technical support"),
    ]
)
```

--------------------------------

### Define Custom Guardrail Agent (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Defines a 'Guardrail check' agent with a Pydantic model 'HomeworkOutput' to determine if a user query is about homework. This agent is used to validate inputs or outputs.

```python
from agents import GuardrailFunctionOutput, Agent, Runner
from pydantic import BaseModel


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)
```

--------------------------------

### List Prompts on Server (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Lists all available prompts on the server. Requires the server session to be initialized by calling `connect()` first. Returns a `ListPromptsResult`.

```python
async def list_prompts(
    self,
) -> ListPromptsResult:
    """List the prompts available on the server."""
    if not self.session:
        raise UserError("Server not initialized. Make sure you call `connect()` first.")

    return await self.session.list_prompts()
```

--------------------------------

### Span Start Method (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Abstract method to initiate the execution of a span. It accepts an optional boolean to mark the span as the current one.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """
    Start the span.

    Args:
        mark_as_current: If true, the span will be marked as the current span.
    """
    pass

```

--------------------------------

### Streaming Hosted MCP Results in Python

Source: https://openai.github.io/openai-agents-python/mcp

This example illustrates how to handle streaming results from a hosted MCP tool. By passing `stream=True` to `Runner.run_streamed`, the code can process incremental output events as the model works, and then accesses the final output.

```python
result = Runner.run_streamed(agent, "Summarise this repository's top languages")
for event in result.stream_events():
    if event.type == "run_item_stream_event":
        print(f"Received: {event.item}")
print(result.final_output)
```

--------------------------------

### on_start Method for Voice Workflow Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/workflow

The async on_start method in VoiceWorkflowBase is an optional hook that runs before any user input is processed. It can be used to deliver introductory messages or instructions via text-to-speech. The default implementation yields nothing, indicating no action is taken.

```python
async def on_start(self) -> AsyncIterator[str]:
    """
    Optional method that runs before any user input is received. Can be used
    to deliver a greeting or instruction via TTS. Defaults to doing nothing.
    """
    return
    yield
```

--------------------------------

### Realtime Session Initialization

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

Initializes a new realtime session with the specified model, agent, and context. Optional model and run configurations can also be provided.

```APIDOC
## RealtimeSession __init__

### Description
Initializes a new realtime session with the specified model, agent, and context. Optional model and run configurations can also be provided.

### Method
__init__

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **model** (RealtimeModel) - Required - The model to use.
- **agent** (RealtimeAgent) - Required - The current agent.
- **context** (TContext | None) - Required - The context object.
- **model_config** (RealtimeModelConfig | None) - Optional - Model configuration. Defaults to None.
- **run_config** (RealtimeRunConfig | None) - Optional - Runtime configuration including guardrails. Defaults to None.

### Request Example
```json
{
  "model": "<RealtimeModel object>",
  "agent": "<RealtimeAgent object>",
  "context": "<TContext object or None>",
  "model_config": "<RealtimeModelConfig object or None>",
  "run_config": "<RealtimeRunConfig object or None>"
}
```

### Response
#### Success Response (None)
This method does not return a value.

#### Response Example
None
```

--------------------------------

### Get or Initialize Async OpenAI Client with Python

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

This Python function `_get_client` ensures that an asynchronous OpenAI client instance is available. If the client (`self._client`) is `None`, it initializes a new `AsyncOpenAI` instance and assigns it. It then returns the client instance, making it ready for use in subsequent API calls.

```python
def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
```

--------------------------------

### RealtimeSession Initialization in Python

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

Initializes a RealtimeSession for real-time model interaction. It requires a model, agent, context, and optional model and run configurations. This class handles the streaming of events and the sending of messages and audio to the model.

```python
class RealtimeSession(RealtimeModelListener):
    """A connection to a realtime model. It streams events from the model to you, and allows you to
    send messages and audio to the model.

    Example:
        ```python
        runner = RealtimeRunner(agent)
        async with await runner.run() as session:
            # Send messages
            await session.send_message("Hello")
            await session.send_audio(audio_bytes)

            # Stream events
            async for event in session:
                if event.type == "audio":
                    # Handle audio event
                    pass
        ```
    """

    def __init__(
        self,
        model: RealtimeModel,
        agent: RealtimeAgent,
        context: TContext | None,
        model_config: RealtimeModelConfig | None = None,
        run_config: RealtimeRunConfig | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            model: The model to use.
            agent: The current agent.
            context: The context object.
            model_config: Model configuration.
            run_config: Runtime configuration including guardrails.
        """
        pass # Placeholder for actual implementation
```

--------------------------------

### Get Text-to-Speech Model

Source: https://openai.github.io/openai-agents-python/zh/ref/voice/models/openai_model_provider

Retrieves a text-to-speech model by its name. If no name is provided, a default model will be returned.

```APIDOC
## GET /api/voice/openai/tts/{model_name}

### Description
Get a text-to-speech model by name.

### Method
GET

### Endpoint
/api/voice/openai/tts/{model_name}

### Parameters
#### Path Parameters
- **model_name** (string) - Required - The name of the model to get.

#### Query Parameters
- **model_name** (string | None) - Optional - The name of the model to get. If not provided, a default model will be used.

### Response
#### Success Response (200)
- **model_id** (string) - The identifier of the text-to-speech model.
- **provider** (string) - The provider of the model (e.g., 'openai').

#### Response Example
```json
{
  "model_id": "tts-1",
  "provider": "openai"
}
```
```

--------------------------------

### Basic Hosted MCP Tool Configuration in Python

Source: https://openai.github.io/openai-agents-python/mcp

This snippet demonstrates how to configure and use a basic hosted MCP tool within an agent. It initializes an Agent with a HostedMCPTool, specifying its configuration including server label and URL. The agent then uses this tool to answer a query.

```python
import asyncio

from agents import Agent, HostedMCPTool, Runner

async def main() -> None:
    agent = Agent(
        name="Assistant",
        tools=[
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "gitmcp",
                    "server_url": "https://gitmcp.io/openai/codex",
                    "require_approval": "never",
                }
            )
        ],
    )

    result = await Runner.run(agent, "Which language is this repository written in?")
    print(result.final_output)

asyncio.run(main())
```

--------------------------------

### Server Initialization Parameters

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Parameters used for initializing the StdioServer.

```APIDOC
## StdioServerParameters

### Description
Parameters for configuring the StdioServer.

### Fields
- **command** (string) - Required - The command to execute.
- **args** (list[string]) - Optional - Arguments for the command.
- **env** (dict[string, string]) - Optional - Environment variables.
- **cwd** (string) - Optional - Current working directory.
- **encoding** (string) - Optional - Encoding for I/O (default: 'utf-8').
- **encoding_error_handler** (string) - Optional - Error handler for encoding (default: 'strict').
```

--------------------------------

### Implement Static Tool Filtering for MCP Server

Source: https://openai.github.io/openai-agents-python/mcp

This example demonstrates static tool filtering for an MCP server using `create_static_tool_filter`. It allows specifying lists of allowed and/or blocked tool names to control which functions the agent can access. The SDK applies the allow-list first, then removes any tools present in the block-list.

```python
from pathlib import Path

from agents.mcp import MCPServerStdio, create_static_tool_filter

samples_dir = Path("/path/to/files")

filesystem_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
    tool_filter=create_static_tool_filter(allowed_tool_names=["read_file", "write_file"]),
)

```

--------------------------------

### Get AsyncOpenAI Client Instance (Python)

Source: https://openai.github.io/openai-agents-python/ref/models/openai_chatcompletions

This method provides a singleton instance of `AsyncOpenAI`. It initializes the client only if it hasn't been created yet, ensuring that a single `AsyncOpenAI` object is reused throughout the application. This is useful for managing API client resources efficiently.

```python
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
```

--------------------------------

### Adding Tools to Realtime Agents (Python)

Source: https://openai.github.io/openai-agents-python/realtime/guide

Demonstrates how to define and add function tools to a RealtimeAgent. These tools allow the agent to perform specific actions during a conversation. The example shows defining `get_weather` and `book_appointment` functions decorated as tools and then passing them to the `RealtimeAgent` constructor.

```python
from agents import function_tool
from agents.realtime import RealtimeAgent

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API logic here
    return f"The weather in {city} is sunny, 72°F"

@function_tool
def book_appointment(date: str, time: str, service: str) -> str:
    """Book an appointment."""
    # Your booking logic here
    return f"Appointment booked for {service} on {date} at {time}"

agent = RealtimeAgent(
    name="Assistant",
    instructions="You can help with weather and appointments.",
    tools=[get_weather, book_appointment],
)
```

--------------------------------

### TracingProcessor Initialization

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processors

Initializes the TracingProcessor with various configuration options for queue size, batch size, schedule delay, and export trigger ratio.

```APIDOC
## TracingProcessor `__init__`

### Description
Initializes the TracingProcessor with configurable parameters for managing and exporting tracing data.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Parameters
- **exporter** (`TracingExporter`) - Required - The exporter to use for sending tracing data.
- **max_queue_size** (`int`) - Optional - The maximum number of spans to store in the queue before dropping spans. Defaults to `8192`.
- **max_batch_size** (`int`) - Optional - The maximum number of spans to export in a single batch. Defaults to `128`.
- **schedule_delay** (`float`) - Optional - The delay in seconds between checks for new spans to export. Defaults to `5.0`.
- **export_trigger_ratio** (`float`) - Optional - The ratio of the queue size at which an export will be triggered. Defaults to `0.7`.

### Request Example
```python
# Example usage (assuming TracingExporter is defined elsewhere)
# from agents.tracing.processors import TracingProcessor
# from agents.tracing.exporters import TracingExporter

# exporter = TracingExporter()
# processor = TracingProcessor(
#     exporter=exporter,
#     max_queue_size=4096,
#     max_batch_size=64,
#     schedule_delay=10.0,
#     export_trigger_ratio=0.8
# )
```

### Response
This is a constructor, so it does not return a value directly.

#### Success Response (200)
N/A

#### Response Example
N/A
```

--------------------------------

### Get Prompt API

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Fetches a specific prompt from the server by its name. Optionally, arguments can be provided to format the prompt.

```APIDOC
## GET /get_prompt

### Description
Retrieves a specific prompt from the server by its name. Arguments can be supplied to customize the prompt content.

### Method
GET

### Endpoint
/get_prompt

### Parameters
#### Query Parameters
- **name** (str) - Required - The name of the prompt to retrieve.
- **arguments** (dict[str, Any]) - Optional - A dictionary of arguments to format the prompt.

### Request Example
```json
{
  "name": "my_custom_prompt",
  "arguments": {
    "user_input": "some text"
  }
}
```

### Response
#### Success Response (200)
- **GetPromptResult** (object) - An object containing the prompt content and metadata.

#### Response Example
```json
{
  "prompt": "This is the content of my_custom_prompt with user input: some text.",
  "variables": ["user_input"]
}
```
```

--------------------------------

### Define Agent Handoffs (Python)

Source: https://openai.github.io/openai-agents-python/quickstart

Defines a 'Triage Agent' that can hand off tasks to other agents ('history_tutor_agent', 'math_tutor_agent') based on user input. This sets up the routing logic for agent orchestration.

```python
from agents import Agent

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)
```

--------------------------------

### Trace Class Abstract Methods in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

This snippet outlines the abstract methods defined in the `Trace` class, which serves as the foundation for creating traceable workflows. It details methods for starting (`start`), finishing (`finish`), entering (`__enter__`), exiting (`__exit__`), and exporting (`export`) trace data. These methods define the interface for managing the lifecycle of a trace and its associated spans.

```python
import abc
from typing import Any

class Trace(abc.ABC):
    """A complete end-to-end workflow containing related spans and metadata."""

    @abc.abstractmethod
    def __enter__(self) -> 'Trace':
        """Enter the runtime context related to this object."""
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """Start the trace and optionally mark it as the current trace."""
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False):
        """Finish the trace and optionally reset the current trace."""
        pass

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """Get the unique identifier for this trace."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the human-readable name of this workflow trace."""
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """Export the trace data as a serializable dictionary."""
        pass

```

--------------------------------

### Trace Management Methods

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/traces

Methods for starting and finishing a trace, with options for managing the current trace context.

```APIDOC
## Trace Management Methods

### Description
Manages the lifecycle of a trace, allowing it to be started and finished. Includes options to control the current trace context.

### Methods

#### `start(mark_as_current: bool = False)`

- **Description**: Starts the trace and optionally marks it as the current trace.
- **Parameters**:
  - `mark_as_current` (bool): If true, marks this trace as the current trace in the execution context. Defaults to `False`.
- **Notes**:
  - Must be called before any spans can be added.
  - Only one trace can be current at a time.
  - Thread-safe when using `mark_as_current`.

#### `finish(reset_current: bool = False)`

- **Description**: Finishes the trace and optionally resets the current trace.
- **Parameters**:
  - `reset_current` (bool): If true, resets the current trace to the previous trace in the execution context. Defaults to `False`.
- **Notes**:
  - Must be called to complete the trace.
  - Finalizes all open spans.
  - Thread-safe when using `reset_current`.
```

--------------------------------

### Create Speech Span - Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

The `speech_span` function creates a new span for speech-related operations. It allows configuration of the speech model, input text, output format, and model parameters. Spans are not started automatically and require manual initiation using `with` statements or `start()`/`finish()` calls. It's part of the tracing capabilities in `src/agents/tracing/create.py`.

```python
def speech_span(
    model: str | None = None,
    input: str | None = None,
    output: str | None = None,
    output_format: str | None = "pcm",
    model_config: Mapping[str, Any] | None = None,
    first_content_at: str | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[SpeechSpanData]:
    """Create a new speech span. The span will not be started automatically, you should either do
    `with speech_span() ...` or call `span.start()` + `span.finish()` manually.

    Args:
        model: The name of the model used for the text-to-speech.
        input: The text input of the text-to-speech.
        output: The audio output of the text-to-speech as base64 encoded string of PCM audio bytes.
        output_format: The format of the audio output (defaults to "pcm").
        model_config: The model configuration (hyperparameters) used.
        first_content_at: The time of the first byte of the audio output.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID.
            We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.
    """
    return get_trace_provider().create_span(
        span_data=SpeechSpanData(
            model=model,
            input=input,
            output=output,
            output_format=output_format,
            model_config=model_config,
            first_content_at=first_content_at,
        ),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )
```

--------------------------------

### Define Computer Creation Protocol (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/tool

Defines the `ComputerCreate` protocol for initializing a computer within a given run context. This protocol specifies a callable that takes a `RunContextWrapper` and returns an awaitable computer instance.

```python
class ComputerCreate(Protocol[ComputerT_co]):
    """Initializes a computer for the current run context."""

    def __call__(self, *, run_context: RunContextWrapper[Any]) -> MaybeAwaitable[ComputerT_co]: ...

```

--------------------------------

### Trace Lifecycle Events API

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

API endpoints for handling the start and end of traces.

```APIDOC
## on_trace_start

### Description
Called when a trace is started.

### Method
POST

### Endpoint
/trace/start

### Parameters
#### Request Body
- **trace** (Trace) - Required - Represents the trace object.

### Request Example
```json
{
  "trace": {
    "trace_id": "string",
    "span_ids": [],
    "start_time": 0,
    "end_time": 0,
    "attributes": {}
  }
}
```

### Response
#### Success Response (200)
- **message** (string) - Indicates successful processing.

#### Response Example
```json
{
  "message": "Trace started successfully."
}
```

## on_trace_end

### Description
Called when a trace is finished.

### Method
POST

### Endpoint
/trace/end

### Parameters
#### Request Body
- **trace** (Trace) - Required - Represents the trace object.

### Request Example
```json
{
  "trace": {
    "trace_id": "string",
    "span_ids": [],
    "start_time": 0,
    "end_time": 0,
    "attributes": {}
  }
}
```

### Response
#### Success Response (200)
- **message** (string) - Indicates successful processing.

#### Response Example
```json
{
  "message": "Trace ended successfully."
}
```
```

--------------------------------

### Initialize OpenAIVoiceModelProvider in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/voice/models/openai_model_provider

This snippet shows how to initialize the OpenAIVoiceModelProvider. It supports direct instantiation with an existing OpenAI client or initialization using API keys, base URLs, organization, and project details. The client is lazily loaded to avoid errors if API keys are not immediately available.

```python
class OpenAIVoiceModelProvider(VoiceModelProvider):
    """A voice model provider that uses OpenAI models."""

    def __init__(
        self,
        *, 
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        """Create a new OpenAI voice model provider.

        Args:
            api_key: The API key to use for the OpenAI client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
                default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            organization: The organization to use for the OpenAI client.
            project: The project to use for the OpenAI client.
        """
        if openai_client is not None:
            assert api_key is None and base_url is None,
                ("Don't provide api_key or base_url if you provide openai_client")
            self._client: AsyncOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project

    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
    # AsyncOpenAI() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=shared_http_client(),
            )

        return self._client
```

--------------------------------

### Time ISO API

Source: https://openai.github.io/openai-agents-python/ref/tracing/provider

Gets the current time formatted in ISO 8601.

```APIDOC
## GET /time/iso

### Description
Return the current time in ISO 8601 format.

### Method
GET

### Endpoint
/time/iso

### Parameters
None

### Request Example
```json
{}
```

### Response
#### Success Response (200)
- **time** (str) - The current time in ISO 8601 format.

#### Response Example
```json
{
  "time": "2023-10-27T10:30:00.123456Z"
}
```
```

--------------------------------

### Get Text-to-Speech Model

Source: https://openai.github.io/openai-agents-python/ko/ref/voice/models/openai_provider

Retrieves a text-to-speech model by its name. If no name is provided, a default model will be returned.

```APIDOC
## get_tts_model

### Description
Get a text-to-speech model by name. If no name is provided, a default model will be returned.

### Method
GET

### Endpoint
/websites/openai_github_io_openai-agents-python/tts_models

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_name** (str | None) - Required - The name of the model to get.

#### Request Body
None

### Request Example
```
GET /websites/openai_github_io_openai-agents-python/tts_models?model_name=tts-1
```

### Response
#### Success Response (200)
- **TTSModel** (TTSModel) - The text-to-speech model.

#### Response Example
```json
{
  "model_type": "OpenAITTSModel",
  "model_name": "tts-1"
}
```
```

--------------------------------

### GET /time_iso

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/provider

Returns the current time formatted as an ISO 8601 string.

```APIDOC
## GET /time_iso

### Description
Return the current time in ISO 8601 format.

### Method
GET

### Endpoint
/time_iso

### Response
#### Success Response (200)
- **current_time** (string) - The current time in ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS.ffffffZ).

#### Response Example
```json
{
  "current_time": "2023-10-27T10:30:00.123456Z"
}
```
```

--------------------------------

### Create Custom Span with Metadata - Python

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

The `custom_span` function creates a new custom span that can include user-defined metadata. Spans are not automatically started; users must explicitly start and finish them using a `with` statement or by calling `span.start()` and `span.finish()` manually. It takes parameters like name, data, span ID, parent, and a disabled flag.

```python
from typing import Any

from agents.tracing.span import Span
from agents.tracing.trace import Trace
from agents.tracing.tracing_provider import get_trace_provider


class CustomSpanData:
    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.data = data

def custom_span(
    name: str,
    data: dict[str, Any] | None = None,
    span_id: str | None = None,
    parent: Trace | Span[Any] | None = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    """Create a new custom span, to which you can add your own metadata. The span will not be
    started automatically, you should either do `with custom_span() ...` or call
    `span.start()` + `span.finish()` manually.

    Args:
        name: The name of the custom span.
        data: Arbitrary structured data to associate with the span.
        span_id: The ID of the span. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are
            correctly formatted.
        parent: The parent span or trace. If not provided, we will automatically use the current
            trace/span as the parent.
        disabled: If True, we will return a Span but the Span will not be recorded.

    Returns:
        The newly created custom span.
    """
    return get_trace_provider().create_span(
        span_data=CustomSpanData(name=name, data=data or {}),
        span_id=span_id,
        parent=parent,
        disabled=disabled,
    )

```

--------------------------------

### Initialize OpenAI Provider (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/multi_provider

Initializes a new OpenAI provider with optional configurations for API key, base URL, client, organization, project, and response usage. It allows for a default provider map or a custom one to be provided.

```python
def __init__(
    self,
    *,
    provider_map: MultiProviderMap | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    openai_organization: str | None = None,
    openai_project: str | None = None,
    openai_use_responses: bool | None = None,
) -> None:
    """Create a new OpenAI provider.

    Args:
        provider_map: A MultiProviderMap that maps prefixes to ModelProviders. If not provided,
            we will use a default mapping. See the documentation for this class to see the
            default mapping.
        openai_api_key: The API key to use for the OpenAI provider. If not provided, we will use
            the default API key.
        openai_base_url: The base URL to use for the OpenAI provider. If not provided, we will
            use the default base URL.
        openai_client: An optional OpenAI client to use. If not provided, we will create a new
            OpenAI client using the api_key and base_url.
        openai_organization: The organization to use for the OpenAI provider.
        openai_project: The project to use for the OpenAI provider.
        openai_use_responses: Whether to use the OpenAI responses API.
    """
    self.provider_map = provider_map
    self.openai_provider = OpenAIProvider(
        api_key=openai_api_key,
        base_url=openai_base_url,
        openai_client=openai_client,
        organization=openai_organization,
        project=openai_project,
        use_responses=openai_use_responses,
    )

    self._fallback_providers: dict[str, ModelProvider] = {}

```

--------------------------------

### Agent Instructions Configuration

Source: https://openai.github.io/openai-agents-python/ja/ref/agent

Configure the instructions for an agent, which serve as the system prompt. Instructions can be a static string or a dynamic callable function that generates them based on context.

```APIDOC
## Agent Instructions Configuration

### Description
The `instructions` attribute defines the behavior and guidelines for an agent. It can be a simple string or a callable function that accepts `context` and `agent` as arguments and returns a string. If a callable is provided, it must accept exactly two parameters.

### Attribute
`instructions`

### Type
```
str | Callable[ [RunContextWrapper[TContext], Agent[TContext]], MaybeAwaitable[str] ] | None
```

### Details
- **String**: A static instruction set for the agent.
- **Callable**: A function that dynamically generates instructions. It receives `run_context` and the `self` (agent instance) and must return a string. This callable can be synchronous or asynchronous.
- **None**: No specific instructions are provided.

### Example (String)
```python
agent.instructions = "You are a helpful assistant."
```

### Example (Callable)
```python
def generate_instructions(context, agent):
    return f"Based on context: {context.get_history()}, assist the user."

agent.instructions = generate_instructions
```

### Example (Async Callable)
```python
async def generate_async_instructions(context, agent):
    history = await context.get_history_async()
    return f"Async instructions based on: {history}"

agent.instructions = generate_async_instructions
```
```

--------------------------------

### Create SQLAlchemy Async Engine and Session

Source: https://openai.github.io/openai-agents-python/sessions

Demonstrates how to create an asynchronous SQLAlchemy engine for PostgreSQL and initialize a SQLAlchemy session. This is suitable for production systems requiring database integration.

```python
from sqlalchemy.ext.asyncio import create_async_engine
from agents.extensions.memory import SQLAlchemySession

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
session = SQLAlchemySession("user_123", engine=engine, create_tables=True)
```

--------------------------------

### GET /gen_group_id

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/provider

Generates and returns a unique group identifier.

```APIDOC
## GET /gen_group_id

### Description
Generate a new group identifier.

### Method
GET

### Endpoint
/gen_group_id

### Response
#### Success Response (200)
- **group_id** (string) - A newly generated unique group identifier.

#### Response Example
```json
{
  "group_id": "group-xyz789"
}
```
```

--------------------------------

### Trace Lifecycle Events API

Source: https://openai.github.io/openai-agents-python/ref/tracing/provider

Methods to handle the start and end of tracing events.

```APIDOC
## on_trace_start

### Description
Called when a trace is started.

### Method
POST (Assumed, as it's a callback for an event)

### Endpoint
/tracing/on_trace_start

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trace** (Trace) - Required - The trace object containing information about the started trace.

### Request Example
```json
{
  "trace": { ... trace object details ... }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing of the event.

#### Response Example
```json
{
  "status": "success"
}
```

## on_trace_end

### Description
Called when a trace is finished.

### Method
POST (Assumed, as it's a callback for an event)

### Endpoint
/tracing/on_trace_end

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trace** (Trace) - Required - The trace object containing information about the finished trace.

### Request Example
```json
{
  "trace": { ... trace object details ... }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing of the event.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### GET /gen_span_id

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/provider

Generates and returns a unique span identifier.

```APIDOC
## GET /gen_span_id

### Description
Generate a new span identifier.

### Method
GET

### Endpoint
/gen_span_id

### Response
#### Success Response (200)
- **span_id** (string) - A newly generated unique span identifier.

#### Response Example
```json
{
  "span_id": "fedcba98-7654-3210-fedc-ba9876543210"
}
```
```

--------------------------------

### Reset Conversation History Wrappers

Source: https://openai.github.io/openai-agents-python/ref/handoffs

Restores the default start and end markers for the conversation history.

```APIDOC
## reset_conversation_history_wrappers

### Description
Restore the default `<CONVERSATION HISTORY>` markers.

### Signature
`reset_conversation_history_wrappers() -> None`

### Source
`src/agents/handoffs/history.py`
```

--------------------------------

### GET /gen_trace_id

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/provider

Generates and returns a unique trace identifier.

```APIDOC
## GET /gen_trace_id

### Description
Generate a new trace identifier.

### Method
GET

### Endpoint
/gen_trace_id

### Response
#### Success Response (200)
- **trace_id** (string) - A newly generated unique trace identifier.

#### Response Example
```json
{
  "trace_id": "98765432-abcd-ef01-2345-6789abcdef01"
}
```
```

--------------------------------

### Exporter Initialization API

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processors

Initializes the tracing exporter with various configuration options including API key, organization, project, endpoint, and retry/delay settings for exponential backoff.

```APIDOC
## __init__

### Description
Initializes the tracing exporter with configuration parameters.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
from agents.tracing.processors import TracingExporter

exporter = TracingExporter(
    api_key="your_api_key",
    organization="your_org_id",
    project="your_project_id",
    endpoint="https://api.openai.com/v1/traces/ingest",
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0
)
```

### Response
#### Success Response (200)
This is a constructor, no direct response.

#### Response Example
N/A
```

--------------------------------

### Initialize MCPServerStdio with Stdio Parameters

Source: https://openai.github.io/openai-agents-python/ref/mcp/server

Initializes the MCPServerStdio, a server implementation using stdio transport. It takes configuration parameters including the command to run, arguments, environment variables, working directory, and text encoding. Optional parameters control tool caching, client session timeouts, tool filtering, structured content usage, and retry mechanisms.

```python
class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server implementation that uses the stdio transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for
    details.
    """

    def __init__(
        self,
        params: MCPServerStdioParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new MCP server based on the stdio transport.

        Args:
            params: The params that configure the server. This includes the command to run to
                start the server, the args to pass to the command, the environment variables to
                set for the server, the working directory to use when spawning the process, and
                the text encoding used when sending/receiving messages to the server.
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).
            name: A readable name for the server. If not provided, we'll create one from the
                command.
            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
        )

        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get("encoding_error_handler", "strict"),
        )

        self._name = name or f"stdio: {self.params.command}"

```

--------------------------------

### Get Prompt

Source: https://openai.github.io/openai-agents-python/ko/ref/mcp/server

Retrieves a specific prompt from the server by its name, with optional arguments.

```APIDOC
## GET /api/prompts/{name}

### Description
Gets a specific prompt from the server by its name. Optional arguments can be provided.

### Method
GET

### Endpoint
/api/prompts/{name}

### Parameters
#### Path Parameters
- **name** (string) - Required - The name of the prompt to retrieve.

#### Query Parameters
- **arguments** (object) - Optional - A dictionary of arguments to pass to the prompt.

### Response
#### Success Response (200)
- **GetPromptResult** (object) - An object containing the prompt details.

#### Response Example
```json
{
  "prompt": "This is the content of the prompt."
}
```
```

--------------------------------

### Python Trace Abstract Methods

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Defines the abstract base class `Trace` in Python, outlining the essential methods and properties for creating and managing traces. This includes methods for entering/exiting the trace context (`__enter__`, `__exit__`), starting and finishing traces (`start`, `finish`), and accessing trace attributes like `trace_id` and `name`. It also includes an abstract method for exporting trace data.

```python
import abc
from typing import Any

class Trace(abc.ABC):
    """A complete end-to-end workflow containing related spans and metadata."""

    @abc.abstractmethod
    def __enter__(self) -> Trace:
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """Start the trace and optionally mark it as the current trace."""
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False):
        """Finish the trace and optionally reset the current trace."""
        pass

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """Get the unique identifier for this trace."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the human-readable name of this workflow trace."""
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """Export the trace data as a serializable dictionary."""
        pass
```

--------------------------------

### Get Default Model

Source: https://openai.github.io/openai-agents-python/ref/models/default_models

Retrieves the name of the default model currently configured.

```APIDOC
## `get_default_model`

### Description
Returns the default model name.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
get_default_model()
```

### Response
#### Success Response (String)
- **model_name** (str) - The name of the default model.

#### Response Example
```json
{
  "model_name": "gpt-4.1"
}
```
```

--------------------------------

### GET /get_tts_model

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/model

Retrieves a text-to-speech model by its specified name.

```APIDOC
## GET /get_tts_model

### Description
Get a text-to-speech model by name.

### Method
GET

### Endpoint
/get_tts_model

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_name** (str | None) - Required - The name of the model to get.

### Request Example
```json
{
  "model_name": "your_model_name"
}
```

### Response
#### Success Response (200)
- **model** (TTSModel) - The text-to-speech model.

#### Response Example
```json
{
  "model": "TTSModel object"
}
```
```

--------------------------------

### Get Specific Prompt

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Fetches a specific prompt from the server by its name. Optional arguments can be provided to customize the prompt's behavior.

```APIDOC
## GET /get_prompt

### Description
Gets a specific prompt from the server by its name. Arguments can be provided to customize the prompt.

### Method
GET

### Endpoint
/get_prompt

### Parameters
#### Query Parameters
- **name** (str) - Required - The name of the prompt to retrieve.
- **arguments** (dict[str, Any]) - Optional - A dictionary of arguments to customize the prompt.

### Request Example
```json
{
  "name": "my_custom_prompt",
  "arguments": {
    "temperature": 0.7,
    "max_tokens": 100
  }
}
```

### Response
#### Success Response (200)
- **prompt_details** (dict) - Contains details about the retrieved prompt, including its content and available options.

#### Response Example
```json
{
  "prompt_details": {
    "content": "This is the content of my_custom_prompt.",
    "options": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }
}
```
```

--------------------------------

### Get System Prompt Async Method (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/agent

Asynchronously retrieves the system prompt for the agent. It checks if instructions are a string or a callable function and returns the appropriate prompt. If instructions are invalid, it logs an error and returns None.

```python
async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
    """Get the system prompt for the agent."""
    if isinstance(self.instructions, str):
        return self.instructions
    elif callable(self.instructions):
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], self.instructions(run_context, self))
        else:
            return cast(str, self.instructions(run_context, self))
    elif self.instructions is not None:
        logger.error(f"Instructions must be a string or a function, got {self.instructions}")

    return None
```

--------------------------------

### GET /get_stt_model

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/model

Retrieves a speech-to-text model by its specified name.

```APIDOC
## GET /get_stt_model

### Description
Get a speech-to-text model by name.

### Method
GET

### Endpoint
/get_stt_model

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_name** (str | None) - Required - The name of the model to get.

### Request Example
```json
{
  "model_name": "your_model_name"
}
```

### Response
#### Success Response (200)
- **model** (STTModel) - The speech-to-text model.

#### Response Example
```json
{
  "model": "STTModel object"
}
```
```

--------------------------------

### Get Speech-to-Text Model

Source: https://openai.github.io/openai-agents-python/ko/ref/voice/models/openai_provider

Retrieves a speech-to-text model by its name. If no name is provided, a default model will be returned.

```APIDOC
## get_stt_model

### Description
Get a speech-to-text model by name. If no name is provided, a default model will be returned.

### Method
GET

### Endpoint
/websites/openai_github_io_openai-agents-python/stt_models

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_name** (str | None) - Required - The name of the model to get.

#### Request Body
None

### Request Example
```
GET /websites/openai_github_io_openai-agents-python/stt_models?model_name=whisper-1
```

### Response
#### Success Response (200)
- **STTModel** (STTModel) - The speech-to-text model.

#### Response Example
```json
{
  "model_type": "OpenAISTTModel",
  "model_name": "whisper-1"
}
```
```

--------------------------------

### RealtimeSession Context Management

Source: https://openai.github.io/openai-agents-python/ref/realtime/session

Manages the lifecycle of a RealtimeSession using an asynchronous context manager. This includes starting the session upon entry and closing it upon exit.

```APIDOC
## RealtimeSession Async Context Manager

### Description
Provides asynchronous context management for the RealtimeSession, automatically handling session startup and shutdown.

### Method
`__aenter__` and `__aexit__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
async with RealtimeSession(
    model=my_model,
    agent=my_agent,
    context=my_context
) as session:
    # Use the session here
    await session.send_message("Hello, AI!")
# Session is automatically closed upon exiting the 'with' block
```

### Response
#### Success Response (200)
- **RealtimeSession** (`RealtimeSession`) - The active session object upon entering the context.

#### Response Example
Upon entering the context, the `__aenter__` method returns the `RealtimeSession` object itself.
```

--------------------------------

### RealtimeSession Initialization and State Management (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Initializes the RealtimeSession with model, agent, context, and configuration. It sets up internal state for history, model settings, event queues, and guardrail tracking. Dependencies include asyncio and the Realtime library's data structures.

```python
self._model = model
        self._current_agent = agent
        self._context_wrapper = RunContextWrapper(context)
        self._event_info = RealtimeEventInfo(context=self._context_wrapper)
        self._history: list[RealtimeItem] = []
        self._model_config = model_config or {}
        self._run_config = run_config or {}
        initial_model_settings = self._model_config.get("initial_model_settings")
        run_config_settings = self._run_config.get("model_settings")
        self._base_model_settings: RealtimeSessionModelSettings = {
            **(run_config_settings or {}),
            **(initial_model_settings or {}),
        }
        self._event_queue: asyncio.Queue[RealtimeSessionEvent] = asyncio.Queue()
        self._closed = False
        self._stored_exception: BaseException | None = None

        # Guardrails state tracking
        self._interrupted_response_ids: set[str] = set()
        self._item_transcripts: dict[str, str] = {}  # item_id -> accumulated transcript
        self._item_guardrail_run_counts: dict[str, int] = {}  # item_id -> run count
        self._debounce_text_length = self._run_config.get("guardrails_settings", {}).get(
            "debounce_text_length", 100
        )

        self._guardrail_tasks: set[asyncio.Task[Any]] = set()
        self._tool_call_tasks: set[asyncio.Task[Any]] = set()
        self._async_tool_calls: bool = bool(self._run_config.get("async_tool_calls", True))
```

--------------------------------

### GET /agents/agent_output/name

Source: https://openai.github.io/openai-agents-python/ja/ref/agent_output

Retrieves the name of the output type. This is a simple getter method.

```APIDOC
## GET /agents/agent_output/name

### Description
Retrieves the name of the output type.

### Method
GET

### Endpoint
/agents/agent_output/name

### Parameters
(No parameters are required for this endpoint)

### Request Body
(No request body is expected for this endpoint)

### Response
#### Success Response (200)
- **name** (string) - The name of the output type.

#### Response Example
```json
{
  "name": "string_output_type"
}
```
```

--------------------------------

### Connect OpenAIRealtimeWebSocketModel (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/openai_realtime

Establishes a real-time WebSocket connection to the OpenAI model. This method handles the initial connection setup, including validating connection parameters and configuring model settings. It ensures that a connection is not already established before proceeding.

```python
async def connect(self, options: RealtimeModelConfig) -> None:
        """Establish a connection to the model and keep it alive."""
        assert self._websocket is None, "Already connected"
        assert self._websocket_task is None, "Already connected"

        model_settings: RealtimeSessionModelSettings = options.get("initial_model_settings", {})

        self._playback_tracker = options.get("playback_tracker", None)

        call_id = options.get("call_id")
        model_name = model_settings.get("model_name")
        if call_id and model_name:
            error_message = (
                "Cannot specify both `call_id` and `model_name` "
                "when attaching to an existing realtime call."
            )
            raise UserError(error_message)

        if model_name:
            self.model = model_name

        self._call_id = call_id
```

--------------------------------

### Get Current Trace API

Source: https://openai.github.io/openai-agents-python/ref/tracing/create

Retrieves the currently active trace, if one exists.

```APIDOC
## GET /trace/current

### Description
Returns the currently active trace object. This is useful for accessing trace information within the current execution context.

### Method
GET

### Endpoint
/trace/current

### Parameters
No parameters required.

### Response
#### Success Response (200)
- **Trace** (Trace | None) - The currently active trace object, or null if no trace is active.

#### Response Example
```json
{
  "trace_object": "..."
}
```
```

--------------------------------

### Optional on_start method for VoiceWorkflowBase - Python

Source: https://openai.github.io/openai-agents-python/ref/voice/workflow

The `on_start` method in `VoiceWorkflowBase` provides an optional hook for executing code before any user interaction. It's an asynchronous generator that can yield initial speech content, such as greetings or instructions.

```python
async def on_start(self) -> AsyncIterator[str]:
    """
    Optional method that runs before any user input is received. Can be used
    to deliver a greeting or instruction via TTS. Defaults to doing nothing.
    """
    return
    yield

```

--------------------------------

### Initialize and Get AsyncOpenAI Client (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_chatcompletions

This method provides a singleton-like pattern for accessing the `AsyncOpenAI` client. It ensures that only one instance of the client is created and reused throughout the application, improving efficiency and resource management. The client is initialized lazily on its first use.

```python
def _get_client(self) -> AsyncOpenAI:
    if self._client is None:
        self._client = AsyncOpenAI()
    return self._client
```

--------------------------------

### GET /conversations/items

Source: https://openai.github.io/openai-agents-python/zh/ref/memory

Retrieves the conversation history for a given session. It supports limiting the number of returned items.

```APIDOC
## GET /conversations/items

### Description
Retrieve the conversation history for this session. Supports fetching all items or the latest N items.

### Method
GET

### Endpoint
/conversations/items

### Parameters
#### Query Parameters
- **limit** (integer) - Optional - Maximum number of items to retrieve. If not specified, retrieves all items. When specified, returns the latest N items in chronological order.

### Request Example
```json
{
  "message": "No request body needed for GET"
}
```

### Response
#### Success Response (200)
- **items** (list[object]) - List of input items representing the conversation history

#### Response Example
```json
{
  "items": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi there!"
    }
  ]
}
```
```

--------------------------------

### Span Lifecycle Events API

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

API endpoints for handling the start and end of spans within a trace.

```APIDOC
## on_span_start

### Description
Called when a span is started.

### Method
POST

### Endpoint
/span/start

### Parameters
#### Request Body
- **span** (Span[Any]) - Required - Represents the span object.

### Request Example
```json
{
  "span": {
    "span_id": "string",
    "trace_id": "string",
    "parent_span_id": "string",
    "name": "string",
    "start_time": 0,
    "end_time": 0,
    "attributes": {}
  }
}
```

### Response
#### Success Response (200)
- **message** (string) - Indicates successful processing.

#### Response Example
```json
{
  "message": "Span started successfully."
}
```

## on_span_end

### Description
Called when a span is finished.

### Method
POST

### Endpoint
/span/end

### Parameters
#### Request Body
- **span** (Span[Any]) - Required - Represents the span object.

### Request Example
```json
{
  "span": {
    "span_id": "string",
    "trace_id": "string",
    "parent_span_id": "string",
    "name": "string",
    "start_time": 0,
    "end_time": 0,
    "attributes": {}
  }
}
```

### Response
#### Success Response (200)
- **message** (string) - Indicates successful processing.

#### Response Example
```json
{
  "message": "Span ended successfully."
}
```
```

--------------------------------

### Get All Function Tools

Source: https://openai.github.io/openai-agents-python/ref/mcp/util

Retrieves all function tools available from a list of MCP servers. It ensures that there are no duplicate tool names across different servers.

```APIDOC
## GET /tools/all

### Description
Get all function tools from a list of MCP servers. Raises a UserError if duplicate tool names are found across servers.

### Method
GET

### Endpoint
/tools/all

### Parameters
#### Query Parameters
- **servers** (list[MCPServer]) - Required - A list of MCP server objects to retrieve tools from.
- **convert_schemas_to_strict** (bool) - Required - Flag to determine if input schemas should be converted to a strict JSON schema.
- **run_context** (RunContextWrapper[Any]) - Required - The runtime context for the operation.
- **agent** (AgentBase) - Required - The agent instance performing the operation.

### Request Example
```json
{
  "servers": [
    {
      "name": "server1",
      "list_tools": "async function() { ... }"
    },
    {
      "name": "server2",
      "list_tools": "async function() { ... }"
    }
  ],
  "convert_schemas_to_strict": true,
  "run_context": "...",
  "agent": "..."
}
```

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of Tool objects representing all function tools found across the servers.

#### Response Example
```json
[
  {
    "name": "tool1",
    "description": "Description of tool1",
    "params_json_schema": { ... },
    "on_invoke_tool": "function() { ... }",
    "strict_json_schema": true
  },
  {
    "name": "tool2",
    "description": "Description of tool2",
    "params_json_schema": { ... },
    "on_invoke_tool": "function() { ... }",
    "strict_json_schema": false
  }
]
```
```

--------------------------------

### Get Trace Provider

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Retrieves the globally configured trace provider used for tracing operations.

```APIDOC
## Get Trace Provider

### Description
Retrieves the global trace provider instance that is used by tracing utilities within the library. This provider is responsible for managing trace and span creation.

### Method
N/A (Function Call)

### Endpoint
N/A (Function Call)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
trace_provider = get_trace_provider()
```

### Response
#### Success Response (200)
- **TraceProvider**: The active global trace provider instance.

#### Response Example
```json
{
  "provider_type": "ExampleTraceProvider"
}
```
```

--------------------------------

### mcp_tools_span

Source: https://openai.github.io/openai-agents-python/ref/tracing/create

Creates a new MCP list tools span. The span must be manually started and finished or used within a `with` statement.

```APIDOC
## POST /mcp_tools_span

### Description
Creates a new MCP list tools span. The span will not be started automatically, you should either do `with mcp_tools_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/mcp_tools_span

### Parameters
#### Request Body
- **server** (str | None) - Optional - The name of the MCP server.
- **result** (list[str] | None) - Optional - The result of the MCP list tools call.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to False.

### Request Example
```json
{
  "server": "mcp_server_1",
  "result": ["tool1", "tool2"],
  "span_id": "optional_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span** (Span[MCPListToolsSpanData]) - The created MCP list tools span.

#### Response Example
```json
{
  "span": {
    "id": "generated_or_provided_span_id",
    "name": "mcp_tools",
    "data": {
      "server": "mcp_server_1",
      "result": ["tool1", "tool2"]
    }
  }
}
```
```

--------------------------------

### Initialize RealtimeSession in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/session

This code demonstrates how to initialize a RealtimeSession, which establishes a connection to a realtime model. It requires the model, agent, and context, with optional configurations for the model and runtime.

```python
class RealtimeSession(RealtimeModelListener):
    """A connection to a realtime model. It streams events from the model to you, and allows you to
    send messages and audio to the model.

    Example:
        ```python
        runner = RealtimeRunner(agent)
        async with await runner.run() as session:
            # Send messages
            await session.send_message("Hello")
            await session.send_audio(audio_bytes)

            # Stream events
            async for event in session:
                if event.type == "audio":
                    # Handle audio event
                    pass
        """

    def __init__(
        self,
        model: RealtimeModel,
        agent: RealtimeAgent,
        context: TContext | None,
        model_config: RealtimeModelConfig | None = None,
        run_config: RealtimeRunConfig | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            model: The model to use.
            agent: The current agent.
            context: The context object.
            model_config: Model configuration.
            run_config: Runtime configuration including guardrails.
        """
        # ... implementation details ...
        pass
```

--------------------------------

### Custom Session Implementation Protocol (Python)

Source: https://openai.github.io/openai-agents-python/sessions

This example shows how to implement a custom session memory by creating a class that adheres to the 'Session' protocol. It outlines the required methods for retrieving, adding, and managing conversation history. Requires the 'agents' library.

```python
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
from typing import List

class MyCustomSession(SessionABC):
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Retrieve conversation history for this session."""
        # Your implementation here
        pass

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Store new items for this session."""
        # Your implementation here
        pass

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from this session."""
        # Your implementation here
        pass

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        # Your implementation here
        pass

# Use your custom session
# agent = Agent(name="Assistant")
# result = await Runner.run(
#     agent,
#     "Hello",
#     session=MyCustomSession("my_session")
# )

```

--------------------------------

### Trace Abstract Base Class Definition in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/traces

This Python code defines the abstract base class `Trace` which outlines the interface for creating and managing traces. It includes abstract methods for entering and exiting the trace context (`__enter__`, `__exit__`), starting and finishing traces (`start`, `finish`), exporting trace data (`export`), and properties for accessing the trace ID (`trace_id`) and name (`name`).

```python
import abc
from typing import Any

class Trace(abc.ABC):
    """A complete end-to-end workflow containing related spans and metadata."""

    @abc.abstractmethod
    def __enter__(self) -> 'Trace':
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False):
        """Start the trace and optionally mark it as the current trace."""
        pass

    @abc.abstractmethod
    def finish(self, reset_current: bool = False):
        """Finish the trace and optionally reset the current trace."""
        pass

    @property
    @abc.abstractmethod
    def trace_id(self) -> str:
        """Get the unique identifier for this trace."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the human-readable name of this workflow trace."""
        pass

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None:
        """Export the trace data as a serializable dictionary."""
        pass

```

--------------------------------

### create_branch_from_turn

Source: https://openai.github.io/openai-agents-python/ko/ref/extensions/memory/advanced_sqlite_session

Creates a new branch in the session, starting from a specific user message turn. This is useful for diverging conversation paths.

```APIDOC
## POST /agents/extensions/memory/advanced_sqlite_session/create_branch_from_turn

### Description
Creates a new branch starting from a specific user message turn. This method allows for diverging conversation paths by creating a new branch based on a historical turn.

### Method
POST

### Endpoint
`/agents/extensions/memory/advanced_sqlite_session/create_branch_from_turn`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **turn_number** (int) - Required - The turn number of the user message to branch from.
- **branch_name** (str | None) - Optional - A name for the new branch. If not provided, a name will be auto-generated.

### Request Example
```json
{
  "turn_number": 10,
  "branch_name": "my_new_branch"
}
```

### Response
#### Success Response (200)
- **branch_id** (str) - The ID of the newly created branch.

#### Response Example
```json
{
  "branch_id": "branch_from_turn_10_1678886400"
}
```

#### Error Response
- **ValueError**: If the specified turn does not exist or does not contain a user message.
```

--------------------------------

### MCPServerSse Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Initializes the MCPServerSse, configuring it for HTTP with SSE transport. This includes setting up server parameters, caching options, timeouts, and retry logic.

```APIDOC
## MCPServerSse

### Description
MCP server implementation that uses the HTTP with SSE transport.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **params** (MCPServerSseParams) - Required - The parameters that configure the server, including URL, headers, and timeouts.
- **cache_tools_list** (bool) - Optional - Whether to cache the tools list. Defaults to False.
- **name** (str | None) - Optional - A readable name for the server. Defaults to a name derived from the URL.
- **client_session_timeout_seconds** (float | None) - Optional - The read timeout for the MCP ClientSession. Defaults to 5.
- **tool_filter** (ToolFilter) - Optional - The tool filter to use.
- **use_structured_content** (bool) - Optional - Whether to use `tool_result.structured_content`. Defaults to False.
- **max_retry_attempts** (int) - Optional - Number of times to retry failed calls. Defaults to 0.
- **retry_backoff_seconds_base** (float) - Optional - The base delay for exponential backoff between retries. Defaults to 1.0.
- **message_handler** (MessageHandlerFnT | None) - Optional - An optional handler invoked for session messages.

### Request Example
```json
{
  "params": {
    "url": "http://example.com/mcp",
    "headers": {"Authorization": "Bearer token"},
    "timeout": 10,
    "sse_read_timeout": 300
  },
  "cache_tools_list": true,
  "client_session_timeout_seconds": 10,
  "use_structured_content": true,
  "max_retry_attempts": 3,
  "retry_backoff_seconds_base": 2.0
}
```

### Response
#### Success Response (200)
Initializes the MCPServerSse object.

#### Response Example
(Initialization does not return a response body in the typical sense, it returns the initialized object.)
```

--------------------------------

### Span Lifecycle Events API

Source: https://openai.github.io/openai-agents-python/ref/tracing/provider

Methods to handle the start and end of individual spans within a trace.

```APIDOC
## on_span_start

### Description
Called when a span is started.

### Method
POST (Assumed, as it's a callback for an event)

### Endpoint
/tracing/on_span_start

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **span** (Span[Any]) - Required - The span object containing information about the started span.

### Request Example
```json
{
  "span": { ... span object details ... }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing of the event.

#### Response Example
```json
{
  "status": "success"
}
```

## on_span_end

### Description
Called when a span is finished.

### Method
POST (Assumed, as it's a callback for an event)

### Endpoint
/tracing/on_span_end

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **span** (Span[Any]) - Required - The span object containing information about the finished span.

### Request Example
```json
{
  "span": { ... span object details ... }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing of the event.

#### Response Example
```json
{
  "status": "success"
}
```
```

--------------------------------

### Format Agent Prompt with Handoff Instructions

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/handoff_prompt

A Python function that takes a user prompt and prepends the recommended system context for agents that support handoffs. This ensures agents are aware of the multi-agent system and handoff mechanisms.

```python
def prompt_with_handoff_instructions(prompt: str) -> str:
    """
    Add recommended instructions to the prompt for agents that use handoffs.
    """
    return f"{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}"
```

--------------------------------

### Span Processing Methods

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processor_interface

Methods called during the lifecycle of a span, including when a span starts and ends.

```APIDOC
## POST /websites/openai_github_io_openai-agents-python/on_span_start

### Description
Called when a new span begins execution. This method is synchronous and should return quickly to avoid blocking execution. Spans are automatically nested under the current trace or span.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/on_span_start

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **span** (Span[Any]) - Required - The span that started. Contains operation details and context.

### Request Example
```json
{
  "span": {
    "operation_name": "example_operation",
    "context": {},
    "start_time": 1678886400.0
  }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing.

#### Response Example
```json
{
  "status": "processed"
}
```

## POST /websites/openai_github_io_openai-agents-python/on_span_end

### Description
Called when a span completes execution. This method is synchronous, should not block or raise exceptions, and is a good time to export or process the individual span.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/on_span_end

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **span** (Span[Any]) - Required - The completed span containing execution results.

### Request Example
```json
{
  "span": {
    "operation_name": "example_operation",
    "context": {},
    "start_time": 1678886400.0,
    "end_time": 1678886460.0,
    "results": {}
  }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing.

#### Response Example
```json
{
  "status": "processed"
}
```
```

--------------------------------

### Configure Agents with Different Models

Source: https://openai.github.io/openai-agents-python/models

Demonstrates how to instantiate `Agent` objects with different models, including specifying model names directly or providing `ModelProvider` implementations. It also shows how to define a `triage_agent` that routes requests to appropriate language-specific agents and an example of running the workflow.

```python
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model="gpt-5-mini", 
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=OpenAIChatCompletionsModel( 
        model="gpt-5-nano",
        openai_client=AsyncOpenAI()
    ),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model="gpt-5",
)

async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


```

--------------------------------

### Trace Processing Methods

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processor_interface

Methods called during the lifecycle of a trace, including when a trace starts and ends.

```APIDOC
## POST /websites/openai_github_io_openai-agents-python/on_trace_start

### Description
Called when a new trace begins execution. This method is synchronous and should return quickly to avoid blocking execution.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/on_trace_start

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trace** (Trace) - Required - The trace that started. Contains workflow name and metadata.

### Request Example
```json
{
  "trace": {
    "workflow_name": "example_workflow",
    "metadata": {}
  }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing.

#### Response Example
```json
{
  "status": "processed"
}
```

## POST /websites/openai_github_io_openai-agents-python/on_trace_end

### Description
Called when a trace completes execution. This method is synchronous and is a good time to export or process the complete trace and handle cleanup of any trace-specific resources.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/on_trace_end

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trace** (Trace) - Required - The completed trace containing all spans and results.

### Request Example
```json
{
  "trace": {
    "workflow_name": "example_workflow",
    "metadata": {},
    "spans": [],
    "results": {}
  }
}
```

### Response
#### Success Response (200)
- **status** (string) - Indicates successful processing.

#### Response Example
```json
{
  "status": "processed"
}
```
```

--------------------------------

### Get Current Span API

Source: https://openai.github.io/openai-agents-python/ref/tracing/create

Retrieves the currently active span, if one exists.

```APIDOC
## GET /span/current

### Description
Returns the currently active span object. This allows access to the current span's details within the active trace.

### Method
GET

### Endpoint
/span/current

### Parameters
No parameters required.

### Response
#### Success Response (200)
- **Span** (Span[Any] | None) - The currently active span object, or null if no span is active.

#### Response Example
```json
{
  "span_object": "..."
}
```
```

--------------------------------

### Define Recommended Prompt Prefix for Handoffs

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/handoff_prompt

Defines a system context prefix that includes instructions for agents operating within the Agents SDK, emphasizing seamless handoffs between agents using a specific function naming convention.

```python
RECOMMENDED_PROMPT_PREFIX = "# System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n"
```

--------------------------------

### Run Interactive Agent REPL Loop (Python)

Source: https://openai.github.io/openai-agents-python/repl

This Python script utilizes the `run_demo_loop` function from the OpenAI Agents SDK to start an interactive chat session directly in the terminal. It initializes an agent with a name and instructions, then facilitates a continuous conversation where the agent remembers history and streams responses. To exit, type 'quit' or 'exit', or use Ctrl-D.

```python
import asyncio
from agents import Agent, run_demo_loop

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)

if __name__ == "__main__":
    asyncio.run(main())

```

--------------------------------

### ComputerCreate Protocol

Source: https://openai.github.io/openai-agents-python/ko/ref/tool

Defines the interface for initializing a computer for the current run context.

```APIDOC
## ComputerCreate Protocol

### Description
Protocol for initializing a computer for the current run context.

### Method
`__call__`

### Parameters
#### Request Body
- **run_context** (RunContextWrapper[Any]) - Required - The run context wrapper.

### Request Example
```python
# Example usage (conceptual)
async def create_my_computer(run_context: RunContextWrapper[Any]) -> MyComputer:
    # ... implementation ...
    pass

computer_creator: ComputerCreate[MyComputer] = create_my_computer
```

### Response
#### Success Response (200)
- **ComputerT_co** (Any) - An instance of the initialized computer.

#### Response Example
```python
# Conceptual representation of a created computer instance
{
  "status": "initialized"
}
```
```

--------------------------------

### Getting Current Trace and Span

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

Retrieve the currently active trace or span using `get_current_trace` and `get_current_span` respectively.

```APIDOC
## GET /tracing/current

### Description

Retrieves the currently active trace and span information.

### Method

GET

### Endpoint

`/tracing/current`

### Responses

#### Success Response (200)

- **trace** (Trace | None) - The current trace object, or None if no trace is active.
- **span** (Span[Any] | None) - The current span object, or None if no span is active.
```

--------------------------------

### Span Lifecycle Methods API

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

This section details the methods for managing the lifecycle of a span, including starting and finishing the operation.

```APIDOC
## Span Lifecycle Methods API

### Description
This section describes the methods used to control the lifecycle of a span.

### Methods

#### `start`
- **Description**: Starts the span.
- **Parameters**:
  - `mark_as_current` (bool, Optional): If true, the span will be marked as the current span. Defaults to `False`.

#### `finish`
- **Description**: Finishes the span.
- **Parameters**:
  - `reset_current` (bool, Optional): If true, the span will be reset as the current span. Defaults to `False`.
- **Returns**: `None`
```

--------------------------------

### OpenAI Conversations API Session Example

Source: https://openai.github.io/openai-agents-python/sessions

Demonstrates using `OpenAIConversationsSession` to leverage OpenAI's Conversations API for managing chat history. This allows for resuming previous conversations by providing a `conversation_id`.

```python
from agents import Agent, Runner, OpenAIConversationsSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a new conversation
session = OpenAIConversationsSession()

# Optionally resume a previous conversation by passing a conversation ID
# session = OpenAIConversationsSession(conversation_id="conv_123")

# Start conversation
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Continue the conversation
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"

```

--------------------------------

### Send User Input via WebSocket in Python

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/openai_realtime

Converts user input into the appropriate event format and sends it over the WebSocket. It then sends a response creation event to signal the start of a model response.

```python
async def _send_user_input(self, event: RealtimeModelSendUserInput) -> None:
    converted = _ConversionHelper.convert_user_input_to_item_create(event)
    await self._send_raw_message(converted)
    await self._send_raw_message(OpenAIResponseCreateEvent(type="response.create"))
```

--------------------------------

### Get ModelProvider by Prefix

Source: https://openai.github.io/openai-agents-python/ko/ref/models/multi_provider

The `get_provider` method retrieves the `ModelProvider` associated with a specific prefix. If the prefix is not found in the map, it returns `None`.

```python
def get_provider(self, prefix: str) -> ModelProvider | None:
    """Returns the ModelProvider for the given prefix.

    Args:
        prefix: The prefix of the model name e.g. "openai" or "my_prefix".
    """
    return self._mapping.get(prefix)
```

--------------------------------

### List Available Prompts

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Retrieves a list of all prompts that are currently available on the server. This function requires the server to be initialized.

```APIDOC
## GET /list_prompts

### Description
Lists the prompts available on the server.

### Method
GET

### Endpoint
/list_prompts

### Parameters
#### Query Parameters
None

### Request Example
```json
{
  "example": "No request body needed"
}
```

### Response
#### Success Response (200)
- **prompts** (List[str]) - A list of prompt names available on the server.

#### Response Example
```json
{
  "prompts": ["prompt1", "prompt2"]
}
```
```

--------------------------------

### MCPServerStdioParams Configuration (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Defines `MCPServerStdioParams`, a `TypedDict` for configuring parameters when starting an MCP server via standard input/output. It mirrors `mcp.client.stdio.StdioServerParameters` and allows specifying the command, arguments, environment variables, working directory, and encoding settings for the server process.

```python
class MCPServerStdioParams(TypedDict):
    """Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another
    import.
    """

    command: str
    """The executable to run to start the server. For example, `python` or `node`."""

    args: NotRequired[list[str]]
    """Command line args to pass to the `command` executable. For example, `['foo.py']` or
    `['server.js', '--port', '8080']`."""

    env: NotRequired[dict[str, str]]
    """The environment variables to set for the server. ."""

    cwd: NotRequired[str | Path]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`."""

    encoding_error_handler: NotRequired[Literal["strict", "ignore", "replace"]]
    """The text encoding error handler. Defaults to `strict`.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """

```

--------------------------------

### GET /session/items

Source: https://openai.github.io/openai-agents-python/ja/ref/memory/session

Retrieves the conversation history for the current session. Supports limiting the number of items returned.

```APIDOC
## GET /session/items

### Description
Retrieve the conversation history for this session.

### Method
GET

### Endpoint
`/session/items`

### Query Parameters
- **limit** (int | None) - Optional - Maximum number of items to retrieve. If None, retrieves all items. When specified, returns the latest N items in chronological order.

### Response
#### Success Response (200)
- **items** (list[TResponseInputItem]) - List of input items representing the conversation history

### Response Example
```json
{
  "items": [
    {
      "type": "user",
      "content": "Hello"
    },
    {
      "type": "agent",
      "content": "Hi there!"
    }
  ]
}
```

### Source Code
`src/agents/memory/session.py` (lines 65-76)
```

--------------------------------

### guardrail_span

Source: https://openai.github.io/openai-agents-python/ref/tracing

Creates a new guardrail span. This span is not started automatically and requires manual start/finish or use within a 'with' statement.

```APIDOC
## guardrail_span

### Description
Create a new guardrail span. The span will not be started automatically, you should either do `with guardrail_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/tracing/spans/guardrail

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **name** (str) - Required - The name of the guardrail.
- **triggered** (bool) - Optional - Whether the guardrail was triggered. Defaults to `False`.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated. Recommended to use `util.gen_span_id()` for correct formatting. Defaults to `None`.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used as the parent. Defaults to `None`.
- **disabled** (bool) - Optional - If True, the Span will not be recorded. Defaults to `False`.

### Request Example
```json
{
  "name": "input_validation",
  "triggered": true,
  "span_id": "span-abc",
  "parent": {
    "id": "trace-123"
  },
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span_id** (str) - The ID of the created guardrail span.
- **name** (str) - The name of the guardrail.
- **start_time** (str) - The start time of the span.
- **end_time** (str) - The end time of the span.

#### Response Example
```json
{
  "span_id": "span-abc",
  "name": "input_validation",
  "start_time": "2023-10-27T10:05:00Z",
  "end_time": "2023-10-27T10:05:05Z"
}
```
```

--------------------------------

### Initialize SQLite Database Schema

Source: https://openai.github.io/openai-agents-python/ja/ref/memory

Sets up the necessary database tables for storing session metadata and messages. It creates 'agent_sessions' and 'agent_messages' tables if they do not already exist, defining their schemas with appropriate columns and constraints for efficient data management.

```python
    def _init_db_for_connection(self, conn: sqlite3.Connection) -> None:
        """Initialize the database schema for a specific connection."""
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

```

--------------------------------

### MultiProvider Class Definition and Initialization (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/multi_provider

Defines the MultiProvider class, inheriting from ModelProvider. Its constructor initializes default or custom provider mappings and sets up the OpenAIProvider with configurable parameters like API key, base URL, and client.

```python
class MultiProvider(ModelProvider):
    """This ModelProvider maps to a Model based on the prefix of the model name. By default, the
    mapping is:
    - "openai/" prefix or no prefix -> OpenAIProvider. e.g. "openai/gpt-4.1", "gpt-4.1"
    - "litellm/" prefix -> LitellmProvider. e.g. "litellm/openai/gpt-4.1"

    You can override or customize this mapping.
    """

    def __init__(
        self,
        *,
        provider_map: MultiProviderMap | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        openai_organization: str | None = None,
        openai_project: str | None = None,
        openai_use_responses: bool | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            provider_map: A MultiProviderMap that maps prefixes to ModelProviders. If not provided,
                we will use a default mapping. See the documentation for this class to see the
                default mapping.
            openai_api_key: The API key to use for the OpenAI provider. If not provided, we will use
                the default API key.
            openai_base_url: The base URL to use for the OpenAI provider. If not provided, we will
                use the default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            openai_organization: The organization to use for the OpenAI provider.
            openai_project: The project to use for the OpenAI provider.
            openai_use_responses: Whether to use the OpenAI responses API.
        """
        self.provider_map = provider_map
        self.openai_provider = OpenAIProvider(
            api_key=openai_api_key,
            base_url=openai_base_url,
            openai_client=openai_client,
            organization=openai_organization,
            project=openai_project,
            use_responses=openai_use_responses,
        )

        self._fallback_providers: dict[str, ModelProvider] = {}

```

--------------------------------

### Start Span Method in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/spans

Abstract method to initiate the execution of a tracing span. It includes an optional parameter to mark the span as the current one being processed, aiding in context propagation within the tracing system.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """
    Start the span.

    Args:
        mark_as_current: If true, the span will be marked as the current span.
    """
    pass
```

--------------------------------

### Create Branch From Turn

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Creates a new branch in the session memory starting from a specific user message turn. This allows for branching conversation history.

```APIDOC
## POST /websites/openai_github_io_openai-agents-python/advanced_sqlite_session#create_branch_from_turn

### Description
Creates a new branch starting from a specific user message turn. This is useful for exploring alternative conversation paths without altering the main history.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/advanced_sqlite_session/create_branch_from_turn

### Parameters
#### Path Parameters
- None

#### Query Parameters
- **turn_number** (int) - Required - The turn number of the user message to branch from.
- **branch_name** (str) - Optional - A name for the new branch. If not provided, it will be auto-generated based on the turn number and timestamp.

#### Request Body
This endpoint does not require a request body.

### Request Example
```json
{
  "turn_number": 5,
  "branch_name": "exploration_branch_1"
}
```

### Response
#### Success Response (200)
- **branch_id** (str) - The unique identifier of the newly created branch.

#### Response Example
```json
{
  "branch_id": "exploration_branch_1"
}
```

#### Error Response
- **ValueError**: If the specified turn does not exist or does not contain a user message.
```

--------------------------------

### GET /websites/openai_github_io_openai-agents-python/get_items

Source: https://openai.github.io/openai-agents-python/ref/memory/sqlite_session

Retrieve the conversation history for a given session. This endpoint allows fetching all items or a specified limit of the latest items.

```APIDOC
## GET /websites/openai_github_io_openai-agents-python/get_items

### Description
Retrieve the conversation history for this session. Supports fetching all items or the latest N items.

### Method
GET

### Endpoint
/websites/openai_github_io_openai-agents-python/get_items

### Parameters
#### Query Parameters
- **limit** (integer) - Optional - Maximum number of items to retrieve. If None, retrieves all items. When specified, returns the latest N items in chronological order.

### Response
#### Success Response (200)
- **items** (list[object]) - List of input items representing the conversation history. Each item is a JSON object.

#### Response Example
```json
{
  "items": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi there!"
    }
  ]
}
```
```

--------------------------------

### GET /usage/turn

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Retrieves usage statistics by turn, providing full JSON token details. If no specific turn number is provided, it returns usage for all turns.

```APIDOC
## GET /usage/turn

### Description
Retrieves usage statistics by turn with full JSON token details. If `user_turn_number` is not specified, it returns usage for all turns within the specified or current branch.

### Method
GET

### Endpoint
`/usage/turn`

### Parameters
#### Query Parameters
- **user_turn_number** (integer) - Optional - Specific turn number to retrieve usage for. If omitted, all turns are returned.
- **branch_id** (string) - Optional - The branch ID to retrieve usage from. If omitted, the current branch is used.

### Request Example
```json
{
  "user_turn_number": 5,
  "branch_id": "main"
}
```

### Response
#### Success Response (200)
- **requests** (integer) - Number of requests made in the turn.
- **input_tokens** (integer) - Total input tokens used.
- **output_tokens** (integer) - Total output tokens used.
- **total_tokens** (integer) - Total tokens used (input + output).
- **input_tokens_details** (object) - Detailed breakdown of input tokens.
- **output_tokens_details** (object) - Detailed breakdown of output tokens.
- **user_turn_number** (integer) - The turn number (only present when retrieving all turns).

#### Response Example (Single Turn)
```json
{
  "requests": 1,
  "input_tokens": 150,
  "output_tokens": 200,
  "total_tokens": 350,
  "input_tokens_details": {
    "prompt_tokens": 100,
    "tool_code_tokens": 50
  },
  "output_tokens_details": {
    "completion_tokens": 200
  }
}
```

#### Response Example (All Turns)
```json
[
  {
    "user_turn_number": 1,
    "requests": 1,
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150,
    "input_tokens_details": {},
    "output_tokens_details": {}
  },
  {
    "user_turn_number": 2,
    "requests": 2,
    "input_tokens": 250,
    "output_tokens": 300,
    "total_tokens": 550,
    "input_tokens_details": {},
    "output_tokens_details": {}
  }
]
```

#### Error Response (404)
- **message** (string) - Error message indicating that the requested resource was not found.
```

--------------------------------

### Tool Events

Source: https://openai.github.io/openai-agents-python/ref/realtime/events

These events are triggered when an agent starts or ends a tool call, providing information about the agent, tool, arguments, and context.

```APIDOC
## Tool Start Event

### Description
An agent is starting a tool call.

### Method
Not applicable (Event-driven)

### Endpoint
Not applicable (Event-driven)

### Parameters
#### Request Body
- **agent** (RealtimeAgent) - Required - The agent that updated.
- **tool** (Tool) - Required - The tool being called.
- **arguments** (str) - Required - The arguments passed to the tool as a JSON string.
- **info** (RealtimeEventInfo) - Required - Common info for all events, such as the context.
- **type** (Literal["tool_start"]) - Required - The event type, fixed to "tool_start".

### Response
#### Success Response (200)
Not applicable (Event-driven)

#### Response Example
```json
{
  "agent": {"id": "agent-123"}, 
  "tool": {"name": "search"},
  "arguments": "{\"query\": \"weather in London\"}",
  "info": {"timestamp": "2023-10-27T10:00:00Z"},
  "type": "tool_start"
}
```

## Tool End Event

### Description
An agent has ended a tool call.

### Method
Not applicable (Event-driven)

### Endpoint
Not applicable (Event-driven)

### Parameters
#### Request Body
- **agent** (RealtimeAgent) - Required - The agent that ended the tool call.
- **tool** (Tool) - Required - The tool that was called.
- **arguments** (str) - Required - The arguments passed to the tool as a JSON string.
- **output** (Any) - Required - The output of the tool call.
- **info** (RealtimeEventInfo) - Required - Common info for all events, such as the context.
- **type** (Literal["tool_end"]) - Required - The event type, fixed to "tool_end".

### Response
#### Success Response (200)
Not applicable (Event-driven)

#### Response Example
```json
{
  "agent": {"id": "agent-123"},
  "tool": {"name": "search"},
  "arguments": "{\"query\": \"weather in London\"}",
  "output": {"temperature": "15°C", "condition": "cloudy"},
  "info": {"timestamp": "2023-10-27T10:01:00Z"},
  "type": "tool_end"
}
```
```

--------------------------------

### GET /api/tools

Source: https://openai.github.io/openai-agents-python/ja/ref/agent

Retrieves a list of all available agent tools. This includes both MCP tools and function tools. The function tools are filtered to include only those that are enabled.

```APIDOC
## GET /api/tools

### Description
Retrieves a list of all available agent tools, including MCP tools and function tools. Only enabled function tools are returned.

### Method
GET

### Endpoint
/api/tools

### Parameters
#### Query Parameters
None

### Request Example
```json
{}
```

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available agent tools. Each tool can be an MCP tool or an enabled function tool.

#### Response Example
```json
{
  "tools": [
    {
      "name": "tool_name_1",
      "description": "description_of_tool_1",
      "type": "mcp"
    },
    {
      "name": "function_tool_1",
      "description": "description_of_function_tool_1",
      "type": "function",
      "is_enabled": true
    }
  ]
}
```
```

--------------------------------

### GET /session/usage

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Retrieves cumulative usage statistics for a session, optionally filtered by a specific branch. This endpoint is useful for monitoring resource consumption.

```APIDOC
## GET /session/usage

### Description
Get cumulative usage for session or specific branch.

### Method
GET

### Endpoint
/session/usage

### Parameters
#### Query Parameters
- **branch_id** (str) - Optional - If provided, only get usage for that branch. If not provided, get usage for all branches.

### Response
#### Success Response (200)
- **requests** (int) - The total number of requests made.
- **input_tokens** (int) - The total number of input tokens used.
- **output_tokens** (int) - The total number of output tokens used.
- **total_tokens** (int) - The total number of tokens used (input + output).
- **total_turns** (int) - The total number of turns in the session or branch.

#### Response Example
```json
{
  "requests": 150,
  "input_tokens": 15000,
  "output_tokens": 20000,
  "total_tokens": 35000,
  "total_turns": 75
}
```

#### Error Response (e.g., 404)
- Returns `null` if no usage data is found for the session or branch.
```

--------------------------------

### Realtime Session Enter Method

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

Manually enters the session as an asynchronous context manager. It is recommended to use the `async with` statement instead.

```APIDOC
## RealtimeSession enter

### Description
Manually enters the session as an asynchronous context manager. It is recommended to use the `async with` statement instead. If this method is used, `close()` must be called manually when done.

### Method
enter (async)

### Endpoint
N/A (Instance method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
session = await realtime_session.enter()
# Use the session here
await session.close()
```

### Response
#### Success Response (200)
- **enter**: RealtimeSession - Returns the session object upon successful entry.

#### Response Example
None
```

--------------------------------

### GET /get_current_trace

Source: https://openai.github.io/openai-agents-python/ref/tracing

Retrieves the currently active trace, if one exists. This is useful for context propagation or debugging.

```APIDOC
## GET /get_current_trace

### Description
Return the currently active trace, if any.

### Method
GET

### Endpoint
/get_current_trace

### Parameters
This endpoint does not accept any parameters.

### Request Example
(No request body needed)

### Response
#### Success Response (200)
- **trace** (Trace | None) - The currently active trace object, or null if no trace is active.

#### Response Example
```json
{
  "trace": {
    "trace_id": "some_trace_id",
    "name": "example_trace",
    "group_id": "some_group_id",
    "start_time": "2023-10-27T10:00:00Z",
    "end_time": "2023-10-27T10:01:00Z",
    "metadata": {},
    "status": "OK"
  }
}
```
```

--------------------------------

### MCPServerStreamableHttp Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Initializes the MCPServerStreamableHttp server with various configuration parameters.

```APIDOC
## MCPServerStreamableHttp

### Description
Implements the MCP server using the Streamable HTTP transport. Refer to the [spec](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) for details.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
This is a constructor, so it takes arguments directly.
- **params** (MCPServerStreamableHttpParams) - Required - Parameters that configure the server, including URL, headers, timeouts, and termination behavior.
- **cache_tools_list** (bool) - Optional - Defaults to `False`. If `True`, the tools list is cached to improve latency.
- **name** (str | None) - Optional - A readable name for the server. If not provided, it's generated from the URL.
- **client_session_timeout_seconds** (float | None) - Optional - Defaults to 5. The read timeout for the MCP ClientSession.
- **tool_filter** (ToolFilter) - Optional - The tool filter to use for filtering tools.
- **use_structured_content** (bool) - Optional - Defaults to `False`. If `True`, uses `tool_result.structured_content`. Set to `True` if the server does not duplicate structured content in `tool_result.content`.
- **max_retry_attempts** (int) - Optional - Defaults to 0. Number of times to retry failed `list_tools`/`call_tool` calls.
- **retry_backoff_seconds_base** (float) - Optional - Defaults to 1.0. The base delay in seconds for exponential backoff between retries.
- **message_handler** (MessageHandlerFnT | None) - Optional - An optional handler invoked for session messages.

### Request Example
```python
# Example usage (assuming MCPServerStreamableHttpParams is defined elsewhere)
params = {
    "url": "http://example.com/mcp",
    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
    "timeout": 10,
    "sse_read_timeout": 300,
    "terminate_on_close": True
}
server = MCPServerStreamableHttp(params=params, cache_tools_list=True)
```

### Response
N/A (This is a constructor)

### Error Handling
Potential errors during initialization may arise from invalid `params` or network issues if the server is contacted immediately.
```

--------------------------------

### Tool Start Event for Realtime Agents (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/events

Defines the RealtimeToolStart dataclass, emitted when an agent is about to call a tool. It includes the agent, the tool being called, the arguments as a JSON string, and common event information. This event is vital for monitoring tool usage and debugging agent behavior.

```python
@dataclass
class RealtimeToolStart:
    """An agent is starting a tool call."""

    agent: RealtimeAgent
    """The agent that updated."""

    tool: Tool
    """The tool being called."""

    arguments: str
    """The arguments passed to the tool as a JSON string."""

    info: RealtimeEventInfo
    """Common info for all events, such as the context."""

    type: Literal["tool_start"] = "tool_start"

```

--------------------------------

### GET /websites/openai_github_io_openai-agents-python/get_state

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/model

Retrieves the current playback state of the realtime model, including the item ID, content index, and elapsed milliseconds.

```APIDOC
## GET /websites/openai_github_io_openai-agents-python/get_state

### Description
Retrieves the current playback state of the realtime model, including the item ID, content index, and elapsed milliseconds.

### Method
GET

### Endpoint
/websites/openai_github_io_openai-agents-python/get_state

### Parameters
None

### Response
#### Success Response (200)
- **current_item_id** (str or null) - The ID of the currently playing item, or null if nothing is playing.
- **current_item_content_index** (int or null) - The index of the current audio content, or null.
- **elapsed_ms** (float or null) - The number of milliseconds played for the current item, or null.

### Response Example
```json
{
  "current_item_id": "audio_clip_1",
  "current_item_content_index": 0,
  "elapsed_ms": 1500.5
}
```
```

--------------------------------

### AdvancedSQLiteSession Initialization

Source: https://openai.github.io/openai-agents-python/sessions/advanced_sqlite_session

Demonstrates how to initialize the AdvancedSQLiteSession with different configurations, including in-memory storage, persistent storage, and custom logging.

```APIDOC
## AdvancedSQLiteSession Initialization

### Description
Initializes the `AdvancedSQLiteSession` with various options for conversation storage and logging.

### Method
__init__

### Parameters
#### Path Parameters
- `session_id` (str) - Required - Unique identifier for the conversation session
- `db_path` (str | Path) - Optional - Path to SQLite database file. Defaults to `:memory:` for in-memory storage.
- `create_tables` (bool) - Optional - Whether to automatically create the advanced tables. Defaults to `False`.
- `logger` (logging.Logger | None) - Optional - Custom logger for the session. Defaults to module logger.

### Request Example
```python
from agents.extensions.memory import AdvancedSQLiteSession

# Basic initialization
session = AdvancedSQLiteSession(
    session_id="my_conversation",
    create_tables=True  # Auto-create advanced tables
)

# With persistent storage
session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="path/to/conversations.db",
    create_tables=True
)

# With custom logger
import logging
logger = logging.getLogger("my_app")
session = AdvancedSQLiteSession(
    session_id="session_456",
    create_tables=True,
    logger=logger
)
```
```

--------------------------------

### Initialize OpenAIResponsesModel

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

Initializes the OpenAIResponsesModel with a specified model, an asynchronous OpenAI client, and an optional flag to indicate if the model was explicitly set.

```python
class OpenAIResponsesModel(Model):
    """
    Implementation of `Model` that uses the OpenAI Responses API.
    """

    def __init__(
        self, 
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
        *, 
        model_is_explicit: bool = True,
    ) -> None:
        self.model = model
        self._model_is_explicit = model_is_explicit
        self._client = openai_client
```

--------------------------------

### Agent with Custom Context

Source: https://openai.github.io/openai-agents-python/agents

Illustrates how to configure an agent with a custom context type. The example defines a `UserContext` dataclass and shows how to specify it as the agent's context type, enabling the agent to access user-specific information and dependencies during its run.

```python
from agents import Agent
from dataclasses import dataclass

@dataclass
class UserContext:
    name: str
    uid: str
    is_pro_user: bool

    async def fetch_purchases() -> list[Purchase]:
        return ...

agent = Agent[UserContext](
    ...,
)

```

--------------------------------

### Response Span API

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Creates a new response span. This span is not started automatically and requires manual start/finish calls or usage within a `with` statement.

```APIDOC
## POST /api/tracing/response_span

### Description
Creates a new response span. The span will not be started automatically, you should either do `with response_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/api/tracing/response_span

### Parameters
#### Request Body
- **response** (Response | None) - Optional - The OpenAI Response object.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to False.

### Request Example
```json
{
  "response": {"data": "<openai_response_object>"},
  "span_id": "generated_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[ResponseSpanData]** - The created span object.

#### Response Example
```json
{
  "span": "<span_object>"
}
```
```

--------------------------------

### Span Methods API

Source: https://openai.github.io/openai-agents-python/ref/tracing

Details the methods available for managing spans, including starting and finishing operations, with options to mark spans as current.

```APIDOC
## Span Methods API

### Description
This section details the methods used to control the lifecycle of a span, from its initiation to its completion.

### Methods

#### `start`
- **Description**: Starts the span. Optionally marks the span as the current span.
- **Parameters**:
  - `mark_as_current` (bool, Optional): If true, the span will be marked as the current span. Defaults to `False`.

#### `finish`
- **Description**: Finishes the span. Optionally resets the current span.
- **Parameters**:
  - `reset_current` (bool, Optional): If true, the span will be reset as the current span. Defaults to `False`.
- **Returns**: `None`
```

--------------------------------

### speech_span

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing/create

Creates a new speech span for tracing text-to-speech operations. The span needs to be manually started and finished or used within a context manager.

```APIDOC
## speech_span

### Description
Create a new speech span. The span will not be started automatically, you should either do `with speech_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
This is a function call, not a traditional HTTP method.

### Endpoint
N/A (Python function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
Parameters are passed as arguments to the function:
- **model** (str | None) - Optional - The name of the model used for the text-to-speech.
- **input** (str | None) - Optional - The text input of the text-to-speech.
- **output** (str | None) - Optional - The audio output of the text-to-speech as base64 encoded string of PCM audio bytes.
- **output_format** (str | None) - Optional - The format of the audio output (defaults to "pcm"). Defaults to 'pcm'.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
- **first_content_at** (str | None) - Optional - The time of the first byte of the audio output.
- **span_id** (str | None) - Optional - The ID of the span. Optional. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to False.

### Request Example
```python
# Using context manager
with speech_span(input="Hello, world!") as span:
    # Perform speech synthesis operations
    pass

# Manual start and finish
span = speech_span(input="Hello, world!")
span.start()
# Perform speech synthesis operations
span.finish()
```

### Response
#### Success Response (Span Object)
- **Span[SpeechSpanData]** - A span object that can be used for tracing.

#### Response Example
```python
# The function returns a Span object, actual content depends on context
span_object = speech_span(input="Example") 
print(type(span_object))
# Output: <class 'openai_agents.tracing.span.Span'> 
```
```

--------------------------------

### OpenAI Voice Model Provider Initialization

Source: https://openai.github.io/openai-agents-python/zh/ref/voice/models/openai_model_provider

Initializes a new OpenAI voice model provider. You can provide an existing OpenAI client or specify API key, base URL, organization, and project details.

```APIDOC
## POST /api/voice/openai

### Description
Create a new OpenAI voice model provider.

### Method
POST

### Endpoint
/api/voice/openai

### Parameters
#### Request Body
- **api_key** (string | None) - Optional - The API key to use for the OpenAI client. If not provided, we will use the default API key.
- **base_url** (string | None) - Optional - The base URL to use for the OpenAI client. If not provided, we will use the default base URL.
- **openai_client** (AsyncOpenAI | None) - Optional - An optional OpenAI client to use. If not provided, we will create a new OpenAI client using the api_key and base_url.
- **organization** (string | None) - Optional - The organization to use for the OpenAI client.
- **project** (string | None) - Optional - The project to use for the OpenAI client.

### Request Example
```json
{
  "api_key": "sk-your-api-key",
  "base_url": "https://api.openai.com/v1",
  "organization": "org-your-organization",
  "project": "proj-your-project"
}
```

### Response
#### Success Response (200)
- **message** (string) - Confirmation message indicating successful initialization.
```

--------------------------------

### Get Turn Usage

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Retrieves the total and detailed input/output token usage for a given turn in the session. This is useful for tracking costs and optimizing token usage.

```APIDOC
## GET /usage

### Description
Retrieves the token usage details for the current session turn. This includes total tokens, as well as detailed breakdowns for input and output.

### Method
GET

### Endpoint
/usage

### Query Parameters
None

### Request Body
None

### Request Example
None

### Response
#### Success Response (200)
- **total_tokens** (integer) - The total number of tokens used in the turn.
- **input_tokens_details** (object) - Details about input tokens.
- **output_tokens_details** (object) - Details about output tokens.

#### Response Example
```json
{
  "total_tokens": 150,
  "input_tokens_details": {
    "type": "prompt",
    "tokens": 100
  },
  "output_tokens_details": {
    "type": "completion",
    "tokens": 50
  }
}
```
```

--------------------------------

### Handle Dynamic Instructions in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/agent

This Python code snippet demonstrates how to handle dynamic agent instructions. It checks if instructions are a string or a callable function. If callable, it inspects the function signature to ensure it accepts exactly two arguments (context and agent). It then calls the function, supporting both synchronous and asynchronous functions, to retrieve the instructions.

```python
import inspect
from typing import Callable, cast, Awaitable

# Assuming RunContextWrapper and Agent types are defined elsewhere
# class RunContextWrapper:
#     pass
# class Agent:
#     pass

def process_instructions(self, run_context, AgentClass):
    if isinstance(self.instructions, str):
        return self.instructions
    elif callable(self.instructions):
        # Inspect the signature of the instructions function
        sig = inspect.signature(self.instructions)
        params = list(sig.parameters.values())

        # Enforce exactly 2 parameters
        if len(params) != 2:
            raise TypeError(
                f"'instructions' callable must accept exactly 2 arguments (context, agent), "
                f"but got {len(params)}: {[p.name for p in params]}"
            )

        # Call the instructions function properly
        if inspect.iscoroutinefunction(self.instructions):
            # Assuming AgentClass is the type of self
            return await cast(Awaitable[str], self.instructions(run_context, AgentClass(self)))
        else:
            return cast(str, self.instructions(run_context, AgentClass(self)))

    else:
        # Handle the case where instructions is None or an invalid type
        # logger.error(...) # Assuming logger is defined
        return None
```

--------------------------------

### transcription_span

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/create

Creates a new span for speech-to-text transcription. The span needs to be manually started and finished or used within a `with` statement.

```APIDOC
## POST /transcription_span

### Description
Create a new transcription span. The span will not be started automatically, you should either do `with transcription_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/transcription_span

### Parameters
#### Query Parameters
- **model** (str | None) - Optional - The name of the model used for the speech-to-text.
- **input** (str | None) - Optional - The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes.
- **input_format** (str | None) - Optional - The format of the audio input (defaults to "pcm"). Default: "pcm"
- **output** (str | None) - Optional - The output of the speech-to-text transcription.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
- **span_id** (str | None) - Optional - The ID of the span. Optional. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Default: False

### Request Example
```json
{
  "model": "whisper-1",
  "input": "/path/to/audio.mp3",
  "input_format": "mp3",
  "output": "transcription.txt",
  "model_config": {
    "temperature": 0.2
  },
  "span_id": "span-12345",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[TranscriptionSpanData]** - The newly created speech-to-text span.

#### Response Example
```json
{
  "span_id": "span-12345",
  "data": {
    "model": "whisper-1",
    "input": "/path/to/audio.mp3",
    "input_format": "mp3",
    "output": "transcription.txt",
    "model_config": {
      "temperature": 0.2
    }
  }
}
```
```

--------------------------------

### Get Specific Prompt from Server (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Retrieves a specific prompt from the server by its name, optionally with arguments. Requires the server session to be initialized by calling `connect()` first. Returns a `GetPromptResult`.

```python
async def get_prompt(
    self, name: str, arguments: dict[str, Any] | None = None
) -> GetPromptResult:
    """Get a specific prompt from the server."""
    if not self.session:
        raise UserError("Server not initialized. Make sure you call `connect()` first.")

    return await self.session.get_prompt(name, arguments)
```

--------------------------------

### GET /get_current_span

Source: https://openai.github.io/openai-agents-python/ref/tracing

Retrieves the currently active span, if one exists. This can be used to access the context of the current operation.

```APIDOC
## GET /get_current_span

### Description
Return the currently active span, if any.

### Method
GET

### Endpoint
/get_current_span

### Parameters
This endpoint does not accept any parameters.

### Request Example
(No request body needed)

### Response
#### Success Response (200)
- **span** (Span[Any] | None) - The currently active span object, or null if no span is active.

#### Response Example
```json
{
  "span": {
    "span_id": "some_span_id",
    "parent_id": "some_parent_id",
    "trace_id": "some_trace_id",
    "name": "example_span",
    "start_time": "2023-10-27T10:00:30Z",
    "end_time": "2023-10-27T10:00:45Z",
    "attributes": {},
    "status": "OK"
  }
}
```
```

--------------------------------

### Initialize MCPServerSse in Python

Source: https://openai.github.io/openai-agents-python/ref/mcp/server

Initializes the MCPServerSse class, setting up an MCP server using HTTP with SSE transport. It takes parameters like server URL, headers, timeouts, and options for tool caching, filtering, and retry mechanisms. The constructor calls the parent class constructor and sets up server-specific attributes.

```python
class MCPServerSse(_MCPServerWithClientSession):
    """MCP server implementation that uses the HTTP with SSE transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse)
    for details.
    """

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new MCP server based on the HTTP with SSE transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, and the
                timeout for the SSE connection.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
        )

        self.params = params
        self._name = name or f"sse: {self.params['url']}"
```

--------------------------------

### MultiProvider Model Retrieval Logic (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/multi_provider

Provides the `get_model` method, which takes a model name, determines its prefix, and uses either a custom provider from `provider_map` or a fallback provider (like OpenAI or Litellm) to retrieve the appropriate model instance.

```python
    def get_model(self, model_name: str | None) -> Model:
        """Returns a Model based on the model name. The model name can have a prefix, ending with
        a "/", which will be used to look up the ModelProvider. If there is no prefix, we will use
        the OpenAI provider.

        Args:
            model_name: The name of the model to get.

        Returns:
            A Model.
        """
        prefix, model_name = self._get_prefix_and_model_name(model_name)

        if prefix and self.provider_map and (provider := self.provider_map.get_provider(prefix)):
            return provider.get_model(model_name)
        else:
            return self._get_fallback_provider(prefix).get_model(model_name)

```

--------------------------------

### GET /current_span

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/provider

Retrieves the currently active span, if one exists. This is useful for context propagation within a trace.

```APIDOC
## GET /current_span

### Description
Return the currently active span, if any.

### Method
GET

### Endpoint
/current_span

### Response
#### Success Response (200)
- **span** (Span[Any] | None) - The current span object, or null if no span is active.

#### Response Example
```json
{
  "span": {
    "span_id": "abcdef0123456789",
    "name": "example_span",
    "start_time": "2023-10-27T10:00:00Z",
    "end_time": "2023-10-27T10:05:00Z",
    "metadata": { "key": "value" }
  }
}
```
```

--------------------------------

### Get Default Trace Processor in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/processors

Retrieves the default trace processor. This processor handles the batching and export of traces and spans to the backend. It returns an instance of BatchTraceProcessor.

```python
def default_processor() -> BatchTraceProcessor:
    """The default processor, which exports traces and spans to the backend in batches."""
    return _global_processor
```

--------------------------------

### Include Handoff Instructions in Agent Prompt (Python)

Source: https://openai.github.io/openai-agents-python/handoffs

Shows how to incorporate recommended handoff instructions into an agent's prompt using `RECOMMENDED_PROMPT_PREFIX`. This helps ensure that Large Language Models (LLMs) properly understand and handle handoff events during agent interactions.

```python
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>."
)
```

--------------------------------

### run_sync classmethod

Source: https://openai.github.io/openai-agents-python/ja/ref/run

Synchronously runs an agent workflow from a specified starting agent and input. This method is suitable for non-asynchronous environments and will loop until a final output is generated or exceptions are raised.

```APIDOC
## POST /run_sync

### Description
Synchronously executes an agent workflow, beginning with a designated agent and an initial input. This method is designed for environments without an active event loop, such as standard Python scripts or non-async web frameworks. The agent operates in a loop, processing input, handling agent outputs, performing handoffs to new agents, and executing tool calls until a terminal output is produced or an exception occurs.

### Method
POST

### Endpoint
/run_sync

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **starting_agent** (Agent[TContext]) - Required - The initial agent to begin the workflow.
- **input** (str | list[TResponseInputItem]) - Required - The first piece of data fed into the agent, which can be a simple string or a list of structured input items.
- **context** (TContext | None) - Optional - A context object to be used throughout the agent run. Defaults to None.
- **max_turns** (int) - Optional - The maximum number of agent invocations (turns) allowed before raising a `MaxTurnsExceeded` exception. Defaults to `DEFAULT_MAX_TURNS`.
- **hooks** (RunHooks[TContext] | None) - Optional - An object providing callbacks for different stages of the agent's execution lifecycle. Defaults to None.
- **run_config** (RunConfig | None) - Optional - Global configuration settings that apply to the entire agent execution. Defaults to None.
- **previous_response_id** (str | None) - Optional - The identifier of a prior response, useful for continuity when using OpenAI's Responses API to avoid re-submitting previous input. Defaults to None.
- **auto_previous_response_id** (bool) - Optional - If true, automatically manages the `previous_response_id` for continuity, especially with OpenAI Responses API. Defaults to False.
- **conversation_id** (str | None) - Optional - The identifier for an existing conversation, allowing the agent to resume or continue a previous dialogue. Defaults to None.
- **session** (Session | None) - Optional - A session object that can be used for managing conversation history automatically. Defaults to None.

### Request Example
```json
{
  "starting_agent": "<agent_object>",
  "input": "What is the weather today?",
  "max_turns": 10,
  "conversation_id": "conv_12345"
}
```

### Response
#### Success Response (200)
- **RunResult** (RunResult) - An object containing the complete outcome of the agent's execution, including all intermediate inputs, guardrail results, and the final output from the last agent. The specific type of the final output can vary due to agent handoffs.

#### Response Example
```json
{
  "output": {
    "type": "agent_output",
    "content": "The weather today is sunny with a high of 75 degrees Fahrenheit."
  },
  "guardrail_results": {},
  "inputs": [
    {
      "type": "user_input",
      "content": "What is the weather today?"
    }
  ],
  "final_agent_id": "weather_agent"
}
```

#### Error Responses
- **MaxTurnsExceeded** - Raised if the agent exceeds the configured `max_turns` limit.
- **GuardrailTripwireTriggered** - Raised if a guardrail's tripwire condition is met during execution.
```

--------------------------------

### Create Custom Span with Data - Python

Source: https://openai.github.io/openai-agents-python/ref/tracing/spans

Demonstrates how to create a custom span for tracing operations. This example shows a database query operation, capturing timing, context, and output data. It utilizes a context manager (`with`) for automatic span management.

```python
from agents.span import custom_span
# Assuming db is an async database client

# Creating a custom span
with custom_span("database_query", {
    "operation": "SELECT",
    "table": "users"
}) as span:
    results = await db.query("SELECT * FROM users")
    span.set_output({"count": len(results)})
```

--------------------------------

### speech_group_span

Source: https://openai.github.io/openai-agents-python/ref/tracing/create

Creates a new speech group span. The span must be manually started and finished or used within a `with` statement.

```APIDOC
## POST /speech_group_span

### Description
Creates a new speech group span. The span will not be started automatically, you should either do `with speech_group_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/speech_group_span

### Parameters
#### Request Body
- **input** (str | None) - Optional - The input text used for the speech request.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to False.

### Request Example
```json
{
  "input": "Hello, how can I help you today?",
  "span_id": "optional_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span** (Span[SpeechGroupSpanData]) - The created speech group span.

#### Response Example
```json
{
  "span": {
    "id": "generated_or_provided_span_id",
    "name": "speech_group",
    "data": {
      "input": "Hello, how can I help you today?"
    }
  }
}
```
```

--------------------------------

### handoff_span

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Creates a new handoff span for tracing agent handoffs. The span needs to be manually started and finished or used within a `with` statement.

```APIDOC
## handoff_span

### Description
Create a new handoff span. The span will not be started automatically, you should either do `with handoff_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
N/A (This is a function call, not a direct HTTP endpoint)

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A (This is a function call with parameters)

- **from_agent** (str | None) - Optional - The name of the agent that is handing off.
- **to_agent** (str | None) - Optional - The name of the agent that is receiving the handoff.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to `False`.

### Request Example
```python
# Example using a with statement
with handoff_span(from_agent="agent_a", to_agent="agent_b") as span:
    # Perform actions related to the handoff
    pass

# Example using manual start/finish
span = handoff_span(from_agent="agent_a", to_agent="agent_b")
span.start()
# Perform actions related to the handoff
span.finish()
```

### Response
#### Success Response
- **Span[HandoffSpanData]** - The newly created handoff span.
```

--------------------------------

### Create Branch From Turn API

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Allows the creation of a new branch in the session, starting from a specific user message turn. This is useful for exploring different conversation paths or correcting previous interactions.

```APIDOC
## POST /websites/openai_github_io_openai-agents-python/advanced_sqlite_session/create_branch_from_turn

### Description
Creates a new branch starting from a specific user message turn. If `branch_name` is not provided, it will be auto-generated with a timestamp.

### Method
POST

### Endpoint
`/websites/openai_github_io_openai-agents-python/advanced_sqlite_session/create_branch_from_turn`

### Parameters
#### Query Parameters
- **turn_number** (int) - Required - The turn number of the user message to branch from.
- **branch_name** (str | None) - Optional - A name for the new branch. If not provided, a name will be auto-generated.

### Request Example
```json
{
  "turn_number": 10,
  "branch_name": "my_new_branch"
}
```

### Response
#### Success Response (200)
- **branch_id** (str) - The ID of the newly created branch.

#### Response Example
```json
{
  "branch_id": "branch_from_turn_10_1678886400"
}
```

#### Error Response (400)
- **error** (str) - Description of the error if the turn does not exist or does not contain a user message.

#### Error Example
```json
{
  "error": "Turn 10 does not contain a user message in branch 'main'"
}
```
```

--------------------------------

### Get Realtime Session Configuration in Python

Source: https://openai.github.io/openai-agents-python/ref/realtime/openai_realtime

This Python method generates the configuration object for a realtime session. It constructs arguments for both audio input and output, incorporating settings for format, noise reduction, transcription, turn detection, voice, and speed. Default settings are used if specific options are not provided in the model_settings. It also handles conditional logic based on whether a call ID exists.

```python
    def _get_session_config(
        self, model_settings: RealtimeSessionModelSettings
    ) -> OpenAISessionCreateRequest:
        """Get the session config."""
        audio_input_args = {}

        if self._call_id:
            audio_input_args["format"] = to_realtime_audio_format(
                model_settings.get("input_audio_format")
            )
        else:
            audio_input_args["format"] = to_realtime_audio_format(
                model_settings.get(
                    "input_audio_format", DEFAULT_MODEL_SETTINGS.get("input_audio_format")
                )
            )

        if "input_audio_noise_reduction" in model_settings:
            audio_input_args["noise_reduction"] = model_settings.get("input_audio_noise_reduction")  # type: ignore[assignment]

        if "input_audio_transcription" in model_settings:
            audio_input_args["transcription"] = model_settings.get("input_audio_transcription")  # type: ignore[assignment]
        else:
            audio_input_args["transcription"] = DEFAULT_MODEL_SETTINGS.get(  # type: ignore[assignment]
                "input_audio_transcription"
            )

        if "turn_detection" in model_settings:
            audio_input_args["turn_detection"] = model_settings.get("turn_detection")  # type: ignore[assignment]
        else:
            audio_input_args["turn_detection"] = DEFAULT_MODEL_SETTINGS.get("turn_detection")  # type: ignore[assignment]

        audio_output_args = {
            "voice": model_settings.get("voice", DEFAULT_MODEL_SETTINGS.get("voice")),
        }

        if self._call_id:
            audio_output_args["format"] = to_realtime_audio_format(  # type: ignore[assignment]
                model_settings.get("output_audio_format")
            )
        else:
            audio_output_args["format"] = to_realtime_audio_format(  # type: ignore[assignment]
                model_settings.get(
                    "output_audio_format", DEFAULT_MODEL_SETTINGS.get("output_audio_format")
                )
            )

        if "speed" in model_settings:
            audio_output_args["speed"] = model_settings.get("speed")  # type: ignore[assignment]

        # Construct full session object. `type` will be excluded at serialization time for updates.
        session_create_request = OpenAISessionCreateRequest(
            type="realtime",
            model=(model_settings.get("model_name") or self.model) or "gpt-realtime",
            output_modalities=model_settings.get(
                "modalities", DEFAULT_MODEL_SETTINGS.get("modalities")
            ),
            audio=OpenAIRealtimeAudioConfig(
                input=OpenAIRealtimeAudioInput(**audio_input_args),  # type: ignore[arg-type]
                output=OpenAIRealtimeAudioOutput(**audio_output_args),  # type: ignore[arg-type]
            ),
            tools=cast(
                Any,
                self._tools_to_session_tools(
                    tools=model_settings.get("tools", []),
                    handoffs=model_settings.get("handoffs", []),
                ),
            ),
        )

        if "instructions" in model_settings:
            session_create_request.instructions = model_settings.get("instructions")

        if "prompt" in model_settings:
            _passed_prompt: Prompt = model_settings["prompt"]
            variables: dict[str, Any] | None = _passed_prompt.get("variables")
            session_create_request.prompt = ResponsePrompt(

```

--------------------------------

### GET /conversations/turns

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Retrieves user turns from a conversation, providing content for browsing and branching decisions. It allows fetching turns from a specific branch or the current branch.

```APIDOC
## GET /conversations/turns

### Description
Retrieves user turns from a conversation, providing content for browsing and branching decisions. It allows fetching turns from a specific branch or the current branch.

### Method
GET

### Endpoint
`/conversations/turns`

### Parameters
#### Query Parameters
- **branch_id** (string) - Optional - Branch to get turns from (current branch if None).

### Response
#### Success Response (200)
- **turns** (list[dict]) - A list of dictionaries, where each dictionary represents a conversation turn and contains the following keys:
  - **turn** (integer) - The turn number in the conversation branch.
  - **content** (string) - The user's message content, truncated to 100 characters.
  - **full_content** (string) - The complete user message content.
  - **timestamp** (string) - The timestamp when the turn was created.
  - **can_branch** (boolean) - Indicates if branching is possible for this turn (always true for user messages).

#### Response Example
```json
{
  "turns": [
    {
      "turn": 1,
      "content": "Hello there...",
      "full_content": "Hello there, how can I help you today?",
      "timestamp": "2023-10-27T10:00:00Z",
      "can_branch": true
    },
    {
      "turn": 2,
      "content": "I need assistance with...",
      "full_content": "I need assistance with setting up a new agent.",
      "timestamp": "2023-10-27T10:05:00Z",
      "can_branch": true
    }
  ]
}
```
```

--------------------------------

### Prepare Tools and Handle Tool Choice for OpenAI API in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/litellm

Prepares a list of tools for an OpenAI API call, converting them to the required format and including any handoff tools. It also determines the tool choice and response format based on model settings. Dependencies include Converter class and _to_dump_compatible function.

```python
        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)
```

--------------------------------

### Span Management Methods API

Source: https://openai.github.io/openai-agents-python/ref/tracing/spans

This section outlines the methods available for managing the lifecycle of a trace span, including starting and finishing the span's execution.

```APIDOC
## Span Management Methods

### Description
Methods to control the execution lifecycle of a tracing span.

### Methods

#### start

##### Description
Start the span.

##### Method
`start`

##### Parameters
- **mark_as_current** (`bool`) - Optional - If true, the span will be marked as the current span. Default: `False`

##### Request Example
```python
span.start(mark_as_current=True)
```

#### finish

##### Description
Finish the span.

##### Method
`finish`

##### Parameters
- **reset_current** (`bool`) - Optional - If true, the span will be reset as the current span. Default: `False`

##### Return Value
- **None**: This method does not return any value.

##### Request Example
```python
span.finish(reset_current=True)
```
```

--------------------------------

### Initialize OpenAI Voice Model Provider (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/voice/models/openai_provider

Initializes the OpenAIVoiceModelProvider with optional OpenAI client configurations. It supports direct client injection or lazy initialization using API keys, base URLs, organization, and project details. Dependencies include the `AsyncOpenAI` client and potentially shared HTTP client configurations.

```python
class OpenAIVoiceModelProvider(VoiceModelProvider):
    """A voice model provider that uses OpenAI models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        """Create a new OpenAI voice model provider.

        Args:
            api_key: The API key to use for the OpenAI client. If not provided, we will use the
                default API key.
            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
                default base URL.
            openai_client: An optional OpenAI client to use. If not provided, we will create a new
                OpenAI client using the api_key and base_url.
            organization: The organization to use for the OpenAI client.
            project: The project to use for the OpenAI client.
        """
        if openai_client is not None:
            assert api_key is None and base_url is None,
                ("Don't provide api_key or base_url if you provide openai_client")
            self._client: AsyncOpenAI | None = openai_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url
            self._stored_organization = organization
            self._stored_project = project

    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
    # AsyncOpenAI() raises an error if you don't have an API key set.
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=shared_http_client(),
            )

        return self._client
```

--------------------------------

### SessionABC - Get Items

Source: https://openai.github.io/openai-agents-python/ref/memory/session

Retrieves the conversation history for a given session. It can fetch all items or a specified number of the latest items.

```APIDOC
## GET /sessions/{session_id}/items

### Description
Retrieves the conversation history for this session.

### Method
GET

### Endpoint
/sessions/{session_id}/items

### Parameters
#### Query Parameters
- **limit** (int | None) - Optional - Maximum number of items to retrieve. If None, retrieves all items. When specified, returns the latest N items in chronological order.

### Response
#### Success Response (200)
- **items** (list[TResponseInputItem]) - A list of input items representing the conversation history.

#### Response Example
```json
{
  "items": [
    {
      "content": "Hello!",
      "role": "user"
    },
    {
      "content": "Hi there! How can I help you?",
      "role": "assistant"
    }
  ]
}
```
```

--------------------------------

### Create Local stdio MCP Server

Source: https://openai.github.io/openai-agents-python/mcp

This code defines how to use `MCPServerStdio` for MCP servers that run as local subprocesses. The SDK manages the spawning and closing of the process and its pipes. This is useful for quick prototypes or when the server exposes a command-line interface. It's configured with the command and arguments to execute.

```python
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

current_dir = Path(__file__).parent
samples_dir = current_dir / "sample_files"

async with MCPServerStdio(
    name="Filesystem Server via npx",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
) as server:
    agent = Agent(
        name="Assistant",
        instructions="Use the files in the sample directory to answer questions.",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, "List the files available to you.")
    print(result.final_output)

```

--------------------------------

### SQLite Session Initialization Options

Source: https://openai.github.io/openai-agents-python/sessions

Shows different ways to initialize an `SQLiteSession`. It covers creating an in-memory database that is lost upon process termination, and a persistent file-based database for long-term storage.

```python
from agents import SQLiteSession

# In-memory database (lost when process ends)
session = SQLiteSession("user_123")

# Persistent file-based database
session = SQLiteSession("user_123", "conversations.db")

# Use the session
# result = await Runner.run(
#     agent, # Assuming agent is defined elsewhere
#     "Hello",
#     session=session
# )

```

--------------------------------

### Agent Start Event Data Structure (Python)

Source: https://openai.github.io/openai-agents-python/ref/realtime/events

Defines the data structure for the RealtimeAgentStartEvent, indicating when a new agent has commenced operations. It includes the agent object and common event information.

```python
@dataclass
class RealtimeAgentStartEvent:
    """A new agent has started."""

    agent: RealtimeAgent
    """The new agent."""

    info: RealtimeEventInfo
    """Common info for all events, such as the context."""

    type: Literal["agent_start"] = "agent_start"

```

--------------------------------

### Get System Prompt Method (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/agent

Asynchronously retrieves the system prompt for the agent. It handles both static string instructions and dynamically generated instructions via synchronous or asynchronous functions.

```python
async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
    """Get the system prompt for the agent."""
    if isinstance(self.instructions, str):
        return self.instructions
    elif callable(self.instructions):
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], self.instructions(run_context, self))
        else:
            return cast(str, self.instructions(run_context, self))
    elif self.instructions is not None:
        logger.error(f"Instructions must be a string or a function, got {self.instructions}")

    return None

```

--------------------------------

### Initialize RealtimeSession in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Initializes a RealtimeSession with a specified model, agent, and optional configurations. It sets up internal states for history, model settings, and event queues, preparing the session for interaction with the real-time model.

```python
def __init__(
    self,
    model: RealtimeModel,
    agent: RealtimeAgent,
    context: TContext | None,
    model_config: RealtimeModelConfig | None = None,
    run_config: RealtimeRunConfig | None = None,
) -> None:
    """Initialize the session.

    Args:
        model: The model to use.
        agent: The current agent.
        context: The context object.
        model_config: Model configuration.
        run_config: Runtime configuration including guardrails.
    """
    self._model = model
    self._current_agent = agent
    self._context_wrapper = RunContextWrapper(context)
    self._event_info = RealtimeEventInfo(context=self._context_wrapper)
    self._history: list[RealtimeItem] = []
    self._model_config = model_config or {}
    self._run_config = run_config or {}
    initial_model_settings = self._model_config.get("initial_model_settings")
    run_config_settings = self._run_config.get("model_settings")
    self._base_model_settings: RealtimeSessionModelSettings = {
        **(run_config_settings or {}),
        **(initial_model_settings or {}),
    }
    self._event_queue: asyncio.Queue[RealtimeSessionEvent] = asyncio.Queue()
    self._closed = False
    self._stored_exception: BaseException | None = None

    # Guardrails state tracking
    self._interrupted_response_ids: set[str] = set()
    self._item_transcripts: dict[str, str] = {}  # item_id -> accumulated transcript
    self._item_guardrail_run_counts: dict[str, int] = {}  # item_id -> run count
    self._debounce_text_length = self._run_config.get("guardrails_settings", {}).get(
        "debounce_text_length", 100
    )

    self._guardrail_tasks: set[asyncio.Task[Any]] = set()
    self._tool_call_tasks: set[asyncio.Task[Any]] = set()
    self._async_tool_calls: bool = bool(self._run_config.get("async_tool_calls", True))
```

--------------------------------

### Create Basic Agent Handoff - Python

Source: https://openai.github.io/openai-agents-python/handoffs

Demonstrates the basic usage of creating a handoff from one agent to another using the `handoff()` function from the Agents SDK. It shows how to initialize agents and assign them to the `handoffs` parameter.

```python
from agents import Agent, handoff

billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")


triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])

```

--------------------------------

### Initialize StreamedAudioResult

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/result

Constructs a new StreamedAudioResult instance, setting up the Text-to-Speech (TTS) model, its settings, and the voice pipeline configuration. It initializes internal buffers, queues, and task management attributes necessary for processing and streaming voice data.

```python
def __init__(
    self,
    tts_model: TTSModel,
    tts_settings: TTSModelSettings,
    voice_pipeline_config: VoicePipelineConfig,
):
    """Create a new `StreamedAudioResult` instance.

    Args:
        tts_model: The TTS model to use.
        tts_settings: The TTS settings to use.
        voice_pipeline_config: The voice pipeline config to use.
    """
    self.tts_model = tts_model
    self.tts_settings = tts_settings
    self.total_output_text = ""
    self.instructions = tts_settings.instructions
    self.text_generation_task: asyncio.Task[Any] | None = None

    self._voice_pipeline_config = voice_pipeline_config
    self._text_buffer = ""
    self._turn_text_buffer = ""
    self._queue: asyncio.Queue[VoiceStreamEvent] = asyncio.Queue()
    self._tasks: list[asyncio.Task[Any]] = []
    self._ordered_tasks: list[
        asyncio.Queue[VoiceStreamEvent | None]
    ] = []  # New: list to hold local queues for each text segment
    self._dispatcher_task: asyncio.Task[Any] | None = (
        None  # Task to dispatch audio chunks in order
    )

    self._done_processing = False
    self._buffer_size = tts_settings.buffer_size
    self._started_processing_turn = False
    self._first_byte_received = False
    self._generation_start_time: str | None = None
    self._completed_session = False
    self._stored_exception: BaseException | None = None
    self._tracing_span: Span[SpeechGroupSpanData] | None = None
```

--------------------------------

### Initialize OpenAI Realtime SIP Model (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/openai_realtime

Connects the OpenAIRealtimeSIPModel to a SIP-originated call. It requires a `call_id` within the provided options to establish the connection. The method then delegates to the superclass's connect method with potentially modified options.

```python
class OpenAIRealtimeSIPModel(OpenAIRealtimeWebSocketModel):
    """Realtime model that attaches to SIP-originated calls using a call ID."""

    async def connect(self, options: RealtimeModelConfig) -> None:
        call_id = options.get("call_id")
        if not call_id:
            raise UserError("OpenAIRealtimeSIPModel requires `call_id` in the model configuration.")

        sip_options = options.copy()
        await super().connect(sip_options)
```

--------------------------------

### Get Output Type Name - Python

Source: https://openai.github.io/openai-agents-python/zh/ref/agent_output

Retrieves the string representation of the output type. This function is straightforward and returns the name associated with the output configuration.

```Python
def name(self) -> str:
    """The name of the output type."""
    return _type_to_str(self.output_type)
```

--------------------------------

### Create Custom Spans for Traceable Operations (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

This example demonstrates how to create custom spans using a context manager for traceable operations in Python. It allows logging operation details, timing, and outputs. The `custom_span` function and `Span` class are prerequisites.

```python
# Creating a custom span
with custom_span("database_query", {
    "operation": "SELECT",
    "table": "users"
}) as span:
    results = await db.query("SELECT * FROM users")
    span.set_output({"count": len(results)})
```

--------------------------------

### Initialize OpenAIChatCompletionsModel in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_chatcompletions

Initializes the OpenAIChatCompletionsModel with a specified chat model and an asynchronous OpenAI client. This setup is crucial for making calls to OpenAI's chat completion endpoints.

```python
class OpenAIChatCompletionsModel(Model):
    def __init__(
        self,
        model: str | ChatModel,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.model = model
        self._client = openai_client
```

--------------------------------

### GET /conversations/turns

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Retrieves the conversation history grouped by user turns for a specified branch. If no branch ID is provided, it defaults to the current branch.

```APIDOC
## GET /conversations/turns

### Description
Retrieves the conversation history grouped by user turns for a specified branch. Each turn is represented by a list of messages, including their type and associated tool name if applicable.

### Method
GET

### Endpoint
`/conversations/turns`

### Parameters
#### Query Parameters
- **branch_id** (string) - Optional - The ID of the branch for which to retrieve the conversation. If not provided, the current branch is used.

### Request Example
```json
{
  "branch_id": "optional_branch_id"
}
```

### Response
#### Success Response (200)
- **turn_number** (integer) - The turn number in the conversation.
- **messages** (array) - A list of messages within the turn.
  - **type** (string) - The type of the message (e.g., 'user', 'tool_call').
  - **tool_name** (string) - The name of the tool used in the message, if applicable.

#### Response Example
```json
{
  "1": [
    {"type": "user", "tool_name": null},
    {"type": "tool_code", "tool_name": "python"}
  ],
  "2": [
    {"type": "assistant", "tool_name": null},
    {"type": "tool_code", "tool_name": "search"}
  ]
}
```
```

--------------------------------

### MCP Tools Span API

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Creates a new MCP list tools span. This span is not started automatically and requires manual start/finish calls or usage within a `with` statement.

```APIDOC
## POST /api/tracing/mcp_tools_span

### Description
Creates a new MCP list tools span. The span will not be started automatically, you should either do `with mcp_tools_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/api/tracing/mcp_tools_span

### Parameters
#### Request Body
- **server** (str | None) - Optional - The name of the MCP server.
- **result** (list[str] | None) - Optional - The result of the MCP list tools call.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to False.

### Request Example
```json
{
  "server": "my_mcp_server",
  "result": ["tool1", "tool2"],
  "span_id": "generated_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[MCPListToolsSpanData]** - The created span object.

#### Response Example
```json
{
  "span": "<span_object>"
}
```
```

--------------------------------

### Create Branch From Turn

Source: https://openai.github.io/openai-agents-python/zh/ref/extensions/memory/advanced_sqlite_session

Creates a new branch in the conversation history starting from a specific user message turn. This allows for diverging conversation paths.

```APIDOC
## POST /api/branches/create_from_turn

### Description
Creates a new branch starting from a specific user message turn. If `branch_name` is not provided, it will be auto-generated with a timestamp. This is useful for exploring alternative conversation flows without affecting the main history.

### Method
POST

### Endpoint
`/api/branches/create_from_turn`

### Parameters
#### Query Parameters
- **turn_number** (integer) - Required - The turn number of the user message to branch from.
- **branch_name** (string) - Optional - A custom name for the new branch. If omitted, a name will be generated automatically.

### Request Example
```json
{
  "turn_number": 5,
  "branch_name": "exploration_branch"
}
```

### Response
#### Success Response (200)
- **branch_id** (string) - The unique identifier of the newly created branch.

#### Response Example
```json
{
  "branch_id": "branch_from_turn_5_1678886400"
}
```

#### Error Response (400)
- **error** (string) - Description of the error, e.g., "Turn does not exist or does not contain a user message."

```json
{
  "error": "Turn 3 does not contain a user message in branch 'main'"
}
```
```

--------------------------------

### Agent Instructions Configuration

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/agent

Defines how agent instructions are set, accepting a string or a callable function for dynamic generation. The function receives context and agent instance, returning a string prompt. If the instructions are neither a string nor a callable, an error is logged.

```python
instructions: (
    str
    | Callable[
        [
            RunContextWrapper[TContext],
            RealtimeAgent[TContext],
        ],
        MaybeAwaitable[str],
    ]
    | None
) = None
```

--------------------------------

### Create and Manage Guardrail Tasks in Python

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

These Python functions manage the execution of output guardrails as asynchronous tasks. `_enqueue_guardrail_task` creates and starts a new asyncio task for running guardrails, adding it to a set for tracking and attaching a callback to handle task completion. `_on_guardrail_task_done` is that callback, responsible for removing completed tasks from the tracking set and propagating any exceptions that occurred during task execution as events.

```python
def _enqueue_guardrail_task(self, text: str, response_id: str) -> None:
        # Runs the guardrails in a separate task to avoid blocking the main loop

        task = asyncio.create_task(self._run_output_guardrails(text, response_id))
        self._guardrail_tasks.add(task)

        # Add callback to remove completed tasks and handle exceptions
        task.add_done_callback(self._on_guardrail_task_done)

def _on_guardrail_task_done(self, task: asyncio.Task[Any]) -> None:
        """Handle completion of a guardrail task."""
        # Remove from tracking set
        self._guardrail_tasks.discard(task)

        # Check for exceptions and propagate as events
        if not task.cancelled():
            exception = task.exception()
            if exception:
                # Create an exception event instead of raising
                asyncio.create_task(
                    # This part is truncated in the original text, but would typically involve sending an event
                )
```

--------------------------------

### SingleAgentVoiceWorkflow on_start method in Python

Source: https://openai.github.io/openai-agents-python/zh/ref/voice/workflow

Implements the optional `on_start` asynchronous method for `SingleAgentVoiceWorkflow`. This method is intended to be executed before any user input is received, potentially for delivering greetings or instructions via text-to-speech. The default behavior is to do nothing.

```python
async def on_start(self) -> AsyncIterator[str]:
    """
    Optional method that runs before any user input is received. Can be used
    to deliver a greeting or instruction via TTS. Defaults to doing nothing.
    """
    return
    yield
```

--------------------------------

### Server Initialization Parameters (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/mcp/server

This snippet shows how server initialization parameters like timeout, SSE read timeout, and terminate on close are retrieved using dictionary get methods with default values. These parameters configure the server's behavior and communication.

```python
timeout=self.params.get("timeout", 5),
 sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
 terminate_on_close=self.params.get("terminate_on_close", True),
```

--------------------------------

### GET /api/usage/turn

Source: https://openai.github.io/openai-agents-python/zh/ref/extensions/memory/advanced_sqlite_session

Retrieves usage statistics by turn with full JSON token details. It can fetch usage for a specific turn or for all turns within a branch.

```APIDOC
## GET /api/usage/turn

### Description
Retrieves usage statistics by turn with full JSON token details. If `user_turn_number` is provided, it returns usage for that specific turn. If `user_turn_number` is `None`, it returns usage for all turns in the specified or current branch.

### Method
GET

### Endpoint
`/api/usage/turn`

### Parameters
#### Query Parameters
- **user_turn_number** (integer) - Optional - Specific turn to get usage for. If `None`, returns all turns.
- **branch_id** (string) - Optional - Branch to get usage from (defaults to the current branch if `None`).

### Request Example
```json
{
  "user_turn_number": 5,
  "branch_id": "main"
}
```

### Response
#### Success Response (200)
- **requests** (integer) - The number of requests made in the turn.
- **input_tokens** (integer) - The total number of input tokens used.
- **output_tokens** (integer) - The total number of output tokens generated.
- **total_tokens** (integer) - The sum of input and output tokens.
- **input_tokens_details** (object) - Detailed breakdown of input token usage (e.g., by model or part).
- **output_tokens_details** (object) - Detailed breakdown of output token usage.
- **user_turn_number** (integer) - The turn number (only present when requesting all turns).

#### Response Example (Single Turn)
```json
{
  "requests": 1,
  "input_tokens": 150,
  "output_tokens": 200,
  "total_tokens": 350,
  "input_tokens_details": {
    "model_A": 150
  },
  "output_tokens_details": {
    "model_B": 200
  }
}
```

#### Response Example (All Turns)
```json
[
  {
    "user_turn_number": 1,
    "requests": 1,
    "input_tokens": 100,
    "output_tokens": 120,
    "total_tokens": 220,
    "input_tokens_details": {
      "model_A": 100
    },
    "output_tokens_details": {
      "model_B": 120
    }
  },
  {
    "user_turn_number": 2,
    "requests": 1,
    "input_tokens": 120,
    "output_tokens": 180,
    "total_tokens": 300,
    "input_tokens_details": {
      "model_A": 120
    },
    "output_tokens_details": {
      "model_B": 180
    }
  }
]
```

### Error Handling
- **404 Not Found**: If the specified `branch_id` or `user_turn_number` does not exist.
- **500 Internal Server Error**: If there is a problem processing the request on the server.
```

--------------------------------

### Get MCP Tools

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/agent

Fetches the available tools from the MCP servers. This endpoint retrieves a list of tools that are specifically managed by the MCP system.

```APIDOC
## GET /api/tools/mcp

### Description
Fetches the available tools from the MCP servers.

### Method
GET

### Endpoint
/api/tools/mcp

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
None

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available MCP tools.

#### Response Example
```json
{
  "tools": [
    {
      "name": "example_mcp_tool",
      "description": "An example tool from MCP",
      "parameters": {},
      "return_direct": false
    }
  ]
}
```
```

--------------------------------

### MCPServerSseParams Constructor

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

This endpoint describes the constructor for creating a new MCP server using HTTP with SSE transport. It outlines the parameters available for configuring the server, including connection details, caching options, and retry mechanisms.

```APIDOC
## MCPServerSseParams Constructor

### Description
Creates a new MCP server based on the HTTP with SSE transport.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **params** (`MCPServerSseParams`) - Required - The parameters that configure the server. This includes the URL of the server, the headers to send to the server, the timeout for the HTTP request, and the timeout for the SSE connection.
- **cache_tools_list** (`bool`) - Optional - Whether to cache the tools list. If `True`, the tools list will be cached and only fetched from the server once. If `False`, the tools list will be fetched from the server on each call to `list_tools()`. The cache can be invalidated by calling `invalidate_tools_cache()`. You should set this to `True` if you know the server will not change its tools list, because it can drastically improve latency (by avoiding a round-trip to the server every time). Default: `False`
- **name** (`str | None`) - Optional - A readable name for the server. If not provided, one will be created from the URL. Default: `None`
- **client_session_timeout_seconds** (`float | None`) - Optional - The read timeout passed to the MCP ClientSession. Default: `5`
- **tool_filter** (`ToolFilter`) - Optional - The tool filter to use for filtering tools. Default: `None`
- **use_structured_content** (`bool`) - Optional - Whether to use `tool_result.structured_content` when calling an MCP tool. Defaults to False for backwards compatibility - most MCP servers still include the structured content in the `tool_result.content`, and using it by default will cause duplicate content. You can set this to True if you know the server will not duplicate the structured content in the `tool_result.content`. Default: `False`
- **max_retry_attempts** (`int`) - Optional - Number of times to retry failed list_tools/call_tool calls. Defaults to no retries. Default: `0`
- **retry_backoff_seconds_base** (`float`) - Optional - The base delay, in seconds, for exponential backoff between retries. Default: `1.0`
- **message_handler** (`MessageHandlerFnT | None`) - Optional - Optional handler invoked for session messages as delivered by the ClientSession. Default: `None`

### Request Example
```json
{
  "params": {
    "url": "http://example.com/api",
    "headers": {"Authorization": "Bearer YOUR_API_KEY"},
    "timeout_seconds": 10
  },
  "cache_tools_list": true,
  "name": "MyAgentServer",
  "client_session_timeout_seconds": 10.0,
  "tool_filter": null,
  "use_structured_content": true,
  "max_retry_attempts": 3,
  "retry_backoff_seconds_base": 0.5,
  "message_handler": null
}
```

### Response
#### Success Response (200)
N/A (Constructor does not return a response)

#### Response Example
N/A
```

--------------------------------

### Convert Tools and Handoffs in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

Converts a list of tools and handoff configurations into a format suitable for API requests. It enforces that only one 'ComputerTool' can be provided and converts individual tools and handoffs using internal helper methods. Returns a ConvertedTools object containing the processed tools and any necessary includes.

```python
class Converter:
    @classmethod
    def convert_tools(
        cls,
        tools: list[Tool],
        handoffs: list[Handoff[Any, Any]],
    ) -> ConvertedTools:
        converted_tools: list[ToolParam] = []
        includes: list[ResponseIncludable] = []

        computer_tools = [tool for tool in tools if isinstance(tool, ComputerTool)]
        if len(computer_tools) > 1:
            raise UserError(f"You can only provide one computer tool. Got {len(computer_tools)}")

        for tool in tools:
            converted_tool, include = cls._convert_tool(tool)
            converted_tools.append(converted_tool)
            if include:
                includes.append(include)

        for handoff in handoffs:
            converted_tools.append(cls._convert_handoff_tool(handoff))

        return ConvertedTools(tools=converted_tools, includes=includes)
```

--------------------------------

### speech_span

Source: https://openai.github.io/openai-agents-python/ref/tracing

Creates a new speech span for tracing text-to-speech operations. The span requires manual start and finish calls or can be used with a `with` statement.

```APIDOC
## speech_span

### Description
Creates a new speech span. The span will not be started automatically, you should either do `with speech_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
Not Applicable (Python function)

### Endpoint
Not Applicable (Python function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
# Example usage within a 'with' statement
with speech_span(model='tts-1', input='Hello world!') as span:
    # Perform speech synthesis operations here
    pass

# Example usage with manual start/finish
span = speech_span(model='tts-1', input='Hello world!')
span.start()
# Perform speech synthesis operations here
span.finish()
```

### Response
#### Success Response
- **Span[SpeechSpanData]** - A span object representing the speech operation.

#### Response Example
```json
{
  "span_id": "generated_or_provided_id",
  "data": {
    "model": "tts-1",
    "input": "Hello world!",
    "output": "base64_encoded_pcm_audio_data",
    "output_format": "pcm",
    "model_config": null,
    "first_content_at": "timestamp"
  }
}
```

### Parameters Details
- **model** (`str | None`) - The name of the model used for the text-to-speech. Defaults to `None`.
- **input** (`str | None`) - The text input of the text-to-speech. Defaults to `None`.
- **output** (`str | None`) - The audio output of the text-to-speech as base64 encoded string of PCM audio bytes. Defaults to `None`.
- **output_format** (`str | None`) - The format of the audio output (defaults to "pcm"). Defaults to `'pcm'`.
- **model_config** (`Mapping[str, Any] | None`) - The model configuration (hyperparameters) used. Defaults to `None`.
- **first_content_at** (`str | None`) - The time of the first byte of the audio output. Defaults to `None`.
- **span_id** (`str | None`) - The ID of the span. Optional. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted. Defaults to `None`.
- **parent** (`Trace | Span[Any] | None`) - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent. Defaults to `None`.
- **disabled** (`bool`) - If True, we will return a Span but the Span will not be recorded. Defaults to `False`.
```

--------------------------------

### Trace Methods

Source: https://openai.github.io/openai-agents-python/ref/tracing

This section details the core methods of the Trace class, including starting, finishing, and exporting trace data, as well as accessing trace identifiers and names.

```APIDOC
## Trace Methods

### `start(mark_as_current: bool = False)`

#### Description
Start the trace and optionally mark it as the current trace.

#### Arguments
- **mark_as_current** (bool) - Optional - If true, marks this trace as the current trace in the execution context.

#### Notes
- Must be called before any spans can be added
- Only one trace can be current at a time
- Thread-safe when using `mark_as_current`

### `finish(reset_current: bool = False)`

#### Description
Finish the trace and optionally reset the current trace.

#### Arguments
- **reset_current** (bool) - Optional - If true, resets the current trace to the previous trace in the execution context.

#### Notes
- Must be called to complete the trace
- Finalizes all open spans
- Thread-safe when using `reset_current`

### `trace_id` (property)

#### Description
Get the unique identifier for this trace.

#### Returns
- **trace_id** (str) - The trace's unique ID in the format 'trace_<32_alphanumeric>'

#### Notes
- IDs are globally unique
- Used to link spans to their parent trace
- Can be used to look up traces in the dashboard

### `name` (property)

#### Description
Get the human-readable name of this workflow trace.

#### Returns
- **name** (str) - The workflow name (e.g., "Customer Service", "Data Processing")

#### Notes
- Should be descriptive and meaningful
- Used for grouping and filtering in the dashboard
- Helps identify the purpose of the trace

### `export() -> dict[str, Any] | None`

#### Description
Export the trace data as a serializable dictionary.

#### Returns
- **dict | None** - Dictionary containing trace data, or None if tracing is disabled.

#### Notes
- Includes all spans and their data
- Used for sending traces to backends
- May include metadata and group ID
```

--------------------------------

### Trace Start Callback - Python

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Handles the event when a new trace begins execution. This method is called synchronously and should return quickly to avoid blocking. It receives the `Trace` object containing workflow information and metadata. Errors should be handled internally.

```python
@abc.abstractmethod
def on_trace_start(self, trace: "Trace") -> None:
    """Called when a new trace begins execution.

    Args:
        trace: The trace that started. Contains workflow name and metadata.

    Notes:
        - Called synchronously on trace start
        - Should return quickly to avoid blocking execution
        - Any errors should be caught and handled internally
    """
    pass

```

--------------------------------

### Set API Key

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/processors

Allows updating the OpenAI API key used by the exporter after initialization. This is useful for dynamically changing credentials or if the key was not provided during initial setup.

```APIDOC
## POST /v1/set_api_key

### Description
Sets or updates the OpenAI API key for the tracing exporter. This key is used for authenticating requests to the OpenAI API.

### Method
POST

### Endpoint
/v1/set_api_key

### Parameters
#### Query Parameters
- **api_key** (str) - Required - The OpenAI API key to use. This is the same key used by the OpenAI Python client.

### Request Example
```json
{
  "api_key": "NEW_API_KEY"
}
```

### Response
#### Success Response (200)
Indicates that the API key has been successfully updated.

#### Response Example
```json
{
  "message": "API key updated successfully."
}
```
```

--------------------------------

### Runner.run() Output

Source: https://openai.github.io/openai-agents-python/results

Explains the structure of the results returned by the `Runner.run` and `Runner.run_sync` methods.

```APIDOC
## Runner.run() Output

### Description
When you call the `Runner.run` or `Runner.run_sync` methods, you receive a `RunResult` object. This object contains detailed information about the execution of the agent run.

### Method
GET (Implicit through `Runner.run` or `Runner.run_sync`)

### Endpoint
N/A (This is a library method, not a REST endpoint)

### Parameters
N/A

### Request Example
```python
# Assuming 'runner' is an instance of Runner
result = runner.run(user_input)
```

### Response
#### Success Response (RunResult)
- **final_output** (Any) - The final output of the last agent that ran. Can be a string or an object of the last agent's output type.
- **inputs_for_next_turn** (list) - A list of inputs suitable for the next agent run, including original input and generated items.
- **last_agent** (Agent) - The last agent that executed.
- **new_items** (list) - A list of `RunItem` objects generated during the run (e.g., MessageOutputItem, HandoffCallItem, ToolCallItem).
- **input_guardrail_results** (GuardrailResults) - Results from input guardrails, if applied.
- **output_guardrail_results** (GuardrailResults) - Results from output guardrails, if applied.
- **raw_responses** (list) - The raw `ModelResponse` objects generated by the LLM.
- **input** (Any) - The original input provided to the `run` method.

#### Response Example
```json
{
  "final_output": "Processed result from the last agent",
  "inputs_for_next_turn": [
    // ... list of RunItem objects ...
  ],
  "last_agent": { /* Agent object details */ },
  "new_items": [
    // ... list of RunItem objects ...
  ],
  "input_guardrail_results": { /* Guardrail results object */ },
  "output_guardrail_results": { /* Guardrail results object */ },
  "raw_responses": [
    // ... list of ModelResponse objects ...
  ],
  "input": "Original user input"
}
```
```

--------------------------------

### Create Trace with Context Management or Manual Start/Finish (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

Defines a `trace` function to create a new trace. This trace can be used as a context manager (e.g., `with trace(...)`) or managed manually by calling `start()` and `finish()` methods. It accepts a workflow name, optional trace and group IDs, and metadata. The `disabled` flag can prevent trace recording.

```python
def trace(
    workflow_name: str,
    trace_id: str | None = None,
    group_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Trace:
    """
    Create a new trace. The trace will not be started automatically; you should either use
    it as a context manager (`with trace(...):`) or call `trace.start()` + `trace.finish()`
    manually.

    In addition to the workflow name and optional grouping identifier, you can provide
    an arbitrary metadata dictionary to attach additional user-defined information to
    the trace.

    Args:
        workflow_name: The name of the logical app or workflow. For example, you might provide
            "code_bot" for a coding agent, or "customer_support_agent" for a customer support agent.
        trace_id: The ID of the trace. Optional. If not provided, we will generate an ID. We
            recommend using `util.gen_trace_id()` to generate a trace ID, to guarantee that IDs are
            correctly formatted.
        group_id: Optional grouping identifier to link multiple traces from the same conversation
            or process. For instance, you might use a chat thread ID.
        metadata: Optional dictionary of additional metadata to attach to the trace.
        disabled: If True, we will return a Trace but the Trace will not be recorded.

    Returns:
        The newly created trace object.
    """
    current_trace = get_trace_provider().get_current_trace()
    if current_trace:
        logger.warning(
            "Trace already exists. Creating a new trace, but this is probably a mistake."
        )

    return get_trace_provider().create_trace(
        name=workflow_name,
        trace_id=trace_id,
        group_id=group_id,
        metadata=metadata,
        disabled=disabled,
    )
```

--------------------------------

### VoicePipeline Initialization and Run Method

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/pipeline

Initializes the VoicePipeline with workflow, STT, and TTS models, and defines the primary 'run' method to process audio input. The 'run' method intelligently dispatches to either single-turn or multi-turn processing based on the input type.

```python
class VoicePipeline:
    """An opinionated voice agent pipeline. It works in three steps:
    1. Transcribe audio input into text.
    2. Run the provided `workflow`, which produces a sequence of text responses.
    3. Convert the text responses into streaming audio output.
    """

    def __init__(
        self,
        *,
        workflow: VoiceWorkflowBase,
        stt_model: STTModel | str | None = None,
        tts_model: TTSModel | str | None = None,
        config: VoicePipelineConfig | None = None,
    ):
        """Create a new voice pipeline.

        Args:
            workflow: The workflow to run. See `VoiceWorkflowBase`.
            stt_model: The speech-to-text model to use. If not provided, a default OpenAI
                model will be used.
            tts_model: The text-to-speech model to use. If not provided, a default OpenAI
                model will be used.
            config: The pipeline configuration. If not provided, a default configuration will be
                used.
        """
        self.workflow = workflow
        self.stt_model = stt_model if isinstance(stt_model, STTModel) else None
        self.tts_model = tts_model if isinstance(tts_model, TTSModel) else None
        self._stt_model_name = stt_model if isinstance(stt_model, str) else None
        self._tts_model_name = tts_model if isinstance(tts_model, str) else None
        self.config = config or VoicePipelineConfig()

    async def run(self, audio_input: AudioInput | StreamedAudioInput) -> StreamedAudioResult:
        """Run the voice pipeline.

        Args:
            audio_input: The audio input to process. This can either be an `AudioInput` instance,
                which is a single static buffer, or a `StreamedAudioInput` instance, which is a
                stream of audio data that you can append to.

        Returns:
            A `StreamedAudioResult` instance. You can use this object to stream audio events and
            play them out.
        """
        if isinstance(audio_input, AudioInput):
            return await self._run_single_turn(audio_input)
        elif isinstance(audio_input, StreamedAudioInput):
            return await self._run_multi_turn(audio_input)
        else:
            raise UserError(f"Unsupported audio input type: {type(audio_input)}")

```

--------------------------------

### Conversation History Wrappers API

Source: https://openai.github.io/openai-agents-python/ref/handoffs

Allows overriding the markers that wrap the generated conversation summary in the OpenAI Agents Python library. You can set custom start and end markers.

```APIDOC
## set_conversation_history_wrappers

### Description
Override the markers that wrap the generated conversation summary. Pass `None` to leave either side unchanged.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/set_conversation_history_wrappers

### Parameters
#### Query Parameters
- **start** (str) - Optional - The new starting marker for the conversation summary.
- **end** (str) - Optional - The new ending marker for the conversation summary.

### Request Example
```json
{
  "start": "<|im_start|>",
  "end": "<|im_end|>"
}
```

### Response
#### Success Response (200)
This endpoint does not return a response body, only a success status code.

#### Response Example
(No response body)
```

--------------------------------

### set_conversation_history_wrappers

Source: https://openai.github.io/openai-agents-python/zh/ref/handoffs

Allows overriding the default markers that wrap the generated conversation summary. You can specify new start and end markers, or pass None to leave one side unchanged.

```APIDOC
## set_conversation_history_wrappers

### Description
Override the markers that wrap the generated conversation summary. Pass `None` to leave either side unchanged.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python

### Parameters
#### Query Parameters
- **start** (str | None) - Optional - The new marker to place at the beginning of the conversation summary.
- **end** (str | None) - Optional - The new marker to place at the end of the conversation summary.

### Request Example
```json
{
  "start": "[START]",
  "end": "[END]"
}
```

### Response
#### Success Response (200)
This function does not return any value upon successful execution (returns None).

#### Response Example
```json
null
```
```

--------------------------------

### GET /conversation/turns

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Retrieves a list of user conversation turns, including their content and timestamps, from a specified branch. This is useful for reviewing conversation history and making branching decisions.

```APIDOC
## GET /conversation/turns

### Description
Retrieves user turns with content for easy browsing and branching decisions. Allows fetching turns from a specific branch or the current one if none is specified.

### Method
GET

### Endpoint
/conversation/turns

### Parameters
#### Query Parameters
- **branch_id** (string) - Optional - Branch to get turns from (current branch if None).

### Request Example
```json
{
  "branch_id": "optional_branch_id"
}
```

### Response
#### Success Response (200)
- **turns** (list[dict]) - A list of dictionaries, where each dictionary represents a conversation turn. Each turn dictionary contains:
  - **turn** (integer) - The turn number in the branch.
  - **content** (string) - A truncated version of the user's message content.
  - **full_content** (string) - The complete user message content.
  - **timestamp** (string) - The time when the turn was created.
  - **can_branch** (boolean) - Indicates if the turn can be branched (always True for user messages).

#### Response Example
```json
{
  "turns": [
    {
      "turn": 1,
      "content": "Hello, how are you?",
      "full_content": "Hello, how are you today?",
      "timestamp": "2023-10-27T10:00:00Z",
      "can_branch": true
    },
    {
      "turn": 2,
      "content": "I'm doing well, thanks!",
      "full_content": "I'm doing well, thanks for asking!",
      "timestamp": "2023-10-27T10:05:00Z",
      "can_branch": true
    }
  ]
}
```
```

--------------------------------

### Manager Multi-Agent System Pattern

Source: https://openai.github.io/openai-agents-python/agents

Demonstrates the 'Manager' pattern for multi-agent systems, where a central agent invokes specialized sub-agents as tools. The example shows a `customer_facing_agent` that uses `booking_agent` and `refund_agent` exposed as tools.

```python
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

customer_facing_agent = Agent(
    name="Customer-facing agent",
    instructions=(
        "Handle all direct user communication. "
        "Call the relevant tools when specialized expertise is needed."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)

```

--------------------------------

### speech_group_span

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Creates a new speech group span. This span is not automatically started and requires manual start/finish calls or use within a `with` statement.

```APIDOC
## speech_group_span

### Description
Create a new speech group span. The span will not be started automatically, you should either do `with speech_group_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
`POST` (conceptual, as this is a function call for creating a span, not a direct HTTP request)

### Endpoint
`N/A` (This is a Python function within a library)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
# Example using a 'with' statement
with speech_group_span(input="Hello world") as span:
    # Perform speech-related operations
    pass

# Example with manual start and finish
span = speech_group_span(input="Hello world")
span.start()
# Perform speech-related operations
span.finish()
```

### Response
#### Success Response (Span Object)
- **span** (Span[SpeechGroupSpanData]) - The created speech group span object.

#### Response Example
```python
# The actual returned object is a Span instance, represented conceptually below
{
  "span_id": "generated_or_provided_id",
  "data": {
    "input": "Hello world"
  },
  "start_time": "timestamp",
  "end_time": "timestamp"
}
```

### Errors
- `TypeError`: If input parameters are of incorrect types.
- `ValueError`: If span_id is incorrectly formatted (if provided).
```

--------------------------------

### LitellmModel Initialization and Response Handling (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/litellm

This Python code defines the LitellmModel class, inheriting from 'Model'. It initializes the model with a specified model name, optional base URL, and API key. The `get_response` method handles fetching responses from LLM APIs via LiteLLM, including tool usage and structured output, and logs the received model responses for debugging.

```python
class LitellmModel(Model):
    """This class enables using any model via LiteLLM. LiteLLM allows you to acess OpenAPI,
    Anthropic, Gemini, Mistral, and many other models.
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,  # unused
        conversation_id: str | None = None,  # unused
        prompt: Any | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
            )

            message: litellm.types.utils.Message | None = None
            first_choice: litellm.types.utils.Choices | None = None
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if isinstance(choice, litellm.types.utils.Choices):
                    first_choice = choice
                    message = first_choice.message

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                if message is not None:
                    logger.debug(
                        f"""LLM resp:\n{ 
                            json.dumps(message.model_dump(), indent=2, ensure_ascii=False)
                        }\n"""
                    )
                else:
                    finish_reason = first_choice.finish_reason if first_choice else "-"
                    logger.debug(f"LLM resp had no message. finish_reason: {finish_reason}")

            if hasattr(response, "usage"):
                response_usage = response.usage
                usage = (

```

--------------------------------

### Run Agent

Source: https://openai.github.io/openai-agents-python/zh/ref/run

Initiates an agent run with specified parameters, including starting agent, input, context, and conversation management options. It handles agent handoffs and returns the result of the last agent's execution.

```APIDOC
## POST /run

### Description
Starts an agent run, processing input through a sequence of agents and tools. It supports advanced features like conversation history management, callbacks for lifecycle events, and global run configurations.

### Method
POST

### Endpoint
/run

### Parameters
#### Request Body
- **starting_agent** (Agent) - Required - The initial agent to start the run.
- **input** (Any) - Required - The input provided to the starting agent.
- **context** (Context) - Optional - Context for the agent run.
- **max_turns** (int) - Optional - Maximum number of turns for the agent run.
- **hooks** (object) - Optional - Callbacks for agent lifecycle events.
- **run_config** (object) - Optional - Global settings for the agent run.
- **previous_response_id** (str) - Optional - ID of the previous response to skip input.
- **auto_previous_response_id** (bool) - Optional - Automatically use the previous response ID.
- **conversation_id** (str) - Optional - ID of the conversation for history management.
- **session** (object) - Optional - Session for automatic conversation history management.

### Response
#### Success Response (200)
- **run_result** (object) - Contains inputs, guardrail results, and the output of the last agent.

#### Response Example
{
  "run_result": {
    "inputs": {},
    "guardrails_output": {},
    "output": {}
  }
}
```

--------------------------------

### Get Turn Usage Statistics

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Retrieves usage statistics for a specific turn or all turns within a session branch. This endpoint provides detailed token usage information.

```APIDOC
## GET /websites/openai_github_io_openai-agents-python/get_turn_usage

### Description
Retrieves usage statistics by turn with full JSON token details. You can specify a particular turn number or retrieve statistics for all turns within a session branch.

### Method
GET

### Endpoint
/websites/openai_github_io_openai-agents-python/get_turn_usage

### Parameters
#### Query Parameters
- **user_turn_number** (int) - Optional - Specific turn to get usage for. If None, returns all turns.
- **branch_id** (str) - Optional - Branch to get usage from (current branch if None).

### Response
#### Success Response (200)
- **requests** (int) - Number of requests made in the turn.
- **input_tokens** (int) - Total input tokens used.
- **output_tokens** (int) - Total output tokens used.
- **total_tokens** (int) - Total tokens used (input + output).
- **input_tokens_details** (dict) - Detailed breakdown of input tokens.
- **output_tokens_details** (dict) - Detailed breakdown of output tokens.
- **user_turn_number** (int) - The turn number (returned when requesting all turns).

#### Response Example (Single Turn)
```json
{
  "requests": 1,
  "input_tokens": 150,
  "output_tokens": 300,
  "total_tokens": 450,
  "input_tokens_details": {
    "model_xyz": 150
  },
  "output_tokens_details": {
    "model_xyz": 300
  }
}
```

#### Response Example (All Turns)
```json
[
  {
    "user_turn_number": 1,
    "requests": 1,
    "input_tokens": 150,
    "output_tokens": 300,
    "total_tokens": 450,
    "input_tokens_details": {
      "model_xyz": 150
    },
    "output_tokens_details": {
      "model_xyz": 300
    }
  },
  {
    "user_turn_number": 2,
    "requests": 2,
    "input_tokens": 200,
    "output_tokens": 400,
    "total_tokens": 600,
    "input_tokens_details": {
      "model_abc": 200
    },
    "output_tokens_details": {
      "model_abc": 400
    }
  }
]
```
```

--------------------------------

### RealtimeSession Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Initializes a new RealtimeSession with the specified model, agent, and context. Optional configurations for the model and runtime can also be provided.

```APIDOC
## RealtimeSession `__init__`

### Description
Initialize the session with the provided model, agent, and context. Optional model and run configurations can be supplied.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **model** (`RealtimeModel`) - Required - The model to use.
- **agent** (`RealtimeAgent`) - Required - The current agent.
- **context** (`TContext | None`) - Required - The context object.
- **model_config** (`RealtimeModelConfig | None`) - Optional - Model configuration. Defaults to None.
- **run_config** (`RealtimeRunConfig | None`) - Optional - Runtime configuration including guardrails. Defaults to None.
```

--------------------------------

### Run Async Realtime Session in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/runner

Starts and returns a realtime session for bidirectional communication with a realtime model. This method is asynchronous and requires an `async with` block for proper session management. It takes optional context and model configuration as input.

```python
async def run(
    self, *, context: TContext | None = None, model_config: RealtimeModelConfig | None = None
) -> RealtimeSession:
    """Start and returns a realtime session.

    Returns:
        RealtimeSession: A session object that allows bidirectional communication with the
        realtime model.

    Example:
        ```python
        runner = RealtimeRunner(agent)
        async with await runner.run() as session:
            await session.send_message("Hello")
            async for event in session:
                print(event)
        ```
    """
    # Create and return the connection
    session = RealtimeSession(
        model=self._model,
        agent=self._starting_agent,
        context=context,
        model_config=model_config,
        run_config=self._config,
    )

    return session

```

--------------------------------

### MCPServerSse: Get Server Name Property

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Provides a readable name for the MCPServerSse instance. This property is derived from the server's URL if a specific name is not provided during initialization.

```python
@property
def name(self) -> str:
        """A readable name for the server."""
        return self._name

```

--------------------------------

### Configure Agent Model with Temperature Settings

Source: https://openai.github.io/openai-agents-python/models

Shows how to configure an `Agent` with specific model settings, such as `temperature`, by passing a `ModelSettings` object during agent instantiation. This allows fine-tuning the model's creativity and randomness.

```python
from agents import Agent, ModelSettings

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.1),
)


```

--------------------------------

### Realtime Session Async Context Management

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

Provides methods for entering and exiting the realtime session as an asynchronous context manager.

```APIDOC
## RealtimeSession __aenter__ and __aexit__

### Description
These methods allow the `RealtimeSession` to be used as an asynchronous context manager. `__aenter__` connects to the model and prepares the session, while `__aexit__` ends the session gracefully.

### Method
__aenter__ (async), __aexit__ (async)

### Endpoint
N/A (Instance methods)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **_exc_type**: Any - Exception type if exiting due to an exception.
- **_exc_val**: Any - Exception value if exiting due to an exception.
- **_exc_tb**: Any - Exception traceback if exiting due to an exception.

### Request Example
```python
async with RealtimeSession(...) as session:
    # Use the session here
    pass
```

### Response
#### Success Response (200)
- **__aenter__**: RealtimeSession - Returns the session object upon successful entry.
- **__aexit__**: None - Exits the session. Returns None.

#### Response Example
None
```

--------------------------------

### Get OpenAI Real-time Session Configuration (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/openai_realtime

Constructs the session configuration object for an OpenAI real-time session based on provided model settings. It determines audio input and output formats, noise reduction, transcription settings, turn detection, voice, and speed, using default values if not specified. This object is then used to create or update the session.

```python
    def _get_session_config(
        self,
        model_settings: RealtimeSessionModelSettings
    ) -> OpenAISessionCreateRequest:
        """Get the session config."""
        audio_input_args = {}

        if self._call_id:
            audio_input_args["format"] = to_realtime_audio_format(
                model_settings.get("input_audio_format")
            )
        else:
            audio_input_args["format"] = to_realtime_audio_format(
                model_settings.get(
                    "input_audio_format", DEFAULT_MODEL_SETTINGS.get("input_audio_format")
                )
            )

        if "input_audio_noise_reduction" in model_settings:
            audio_input_args["noise_reduction"] = model_settings.get("input_audio_noise_reduction")  # type: ignore[assignment]

        if "input_audio_transcription" in model_settings:
            audio_input_args["transcription"] = model_settings.get("input_audio_transcription")  # type: ignore[assignment]
        else:
            audio_input_args["transcription"] = DEFAULT_MODEL_SETTINGS.get(  # type: ignore[assignment]
                "input_audio_transcription"
            )

        if "turn_detection" in model_settings:
            audio_input_args["turn_detection"] = model_settings.get("turn_detection")  # type: ignore[assignment]
        else:
            audio_input_args["turn_detection"] = DEFAULT_MODEL_SETTINGS.get("turn_detection")  # type: ignore[assignment]

        audio_output_args = {
            "voice": model_settings.get("voice", DEFAULT_MODEL_SETTINGS.get("voice")),
        }

        if self._call_id:
            audio_output_args["format"] = to_realtime_audio_format(  # type: ignore[assignment]
                model_settings.get("output_audio_format")
            )
        else:
            audio_output_args["format"] = to_realtime_audio_format(  # type: ignore[assignment]
                model_settings.get(
                    "output_audio_format", DEFAULT_MODEL_SETTINGS.get("output_audio_format")
                )
            )

        if "speed" in model_settings:
            audio_output_args["speed"] = model_settings.get("speed")  # type: ignore[assignment]

        # Construct full session object. `type` will be excluded at serialization time for updates.
        session_create_request = OpenAISessionCreateRequest(
            type="realtime",
            model=(model_settings.get("model_name") or self.model) or "gpt-realtime",
            output_modalities=model_settings.get(
                "modalities", DEFAULT_MODEL_SETTINGS.get("modalities")
            ),
            audio=OpenAIRealtimeAudioConfig(
                input=OpenAIRealtimeAudioInput(**audio_input_args),  # type: ignore[arg-type]
                output=OpenAIRealtimeAudioOutput(**audio_output_args),  # type: ignore[arg-type]
            ),
            tools=cast(
                Any,
                self._tools_to_session_tools(
                    tools=model_settings.get("tools", []),
                    handoffs=model_settings.get("handoffs", []),
                ),
            ),
        )

        if "instructions" in model_settings:
            session_create_request.instructions = model_settings.get("instructions")

        if "prompt" in model_settings:
            _passed_prompt: Prompt = model_settings["prompt"]
            variables: dict[str, Any] | None = _passed_prompt.get("variables")
            session_create_request.prompt = ResponsePrompt(

```

--------------------------------

### Get Speech-to-Text Model (Python)

Source: https://openai.github.io/openai-agents-python/ref/voice/models/openai_model_provider

Retrieves a Speech-to-Text (STT) model from the OpenAI provider. If no model name is specified, it defaults to `DEFAULT_STT_MODEL`. This function requires an initialized OpenAI client.

```python
def get_stt_model(self, model_name: str | None) -> STTModel:
    """Get a speech-to-text model by name.

    Args:
        model_name: The name of the model to get.

    Returns:
        The speech-to-text model.
    """
    return OpenAISTTModel(model_name or DEFAULT_STT_MODEL, self._get_client())
```

--------------------------------

### Tracing Processor Initialization (__init__)

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing/processors

Initializes the tracing processor with configuration options for API key, organization, project, endpoint, and retry/delay settings for exponential backoff.

```APIDOC
## POST /v1/traces/ingest

### Description
Initializes the tracing processor with optional API key, organization, project, and custom endpoint. It also configures retry mechanisms with exponential backoff and jitter.

### Method
POST

### Endpoint
/v1/traces/ingest

### Parameters
#### Query Parameters
- **api_key** (str | None) - Optional - The API key for the "Authorization" header. Defaults to `os.environ["OPENAI_API_KEY"]` if not provided.
- **organization** (str | None) - Optional - The OpenAI organization to use. Defaults to `os.environ["OPENAI_ORG_ID"]` if not provided.
- **project** (str | None) - Optional - The OpenAI project to use. Defaults to `os.environ["OPENAI_PROJECT_ID"]` if not provided.
- **endpoint** (str) - Optional - The HTTP endpoint to which traces/spans are posted. Defaults to `'https://api.openai.com/v1/traces/ingest'`.
- **max_retries** (int) - Optional - Maximum number of retries upon failures. Defaults to `3`.
- **base_delay** (float) - Optional - Base delay (in seconds) for the first backoff. Defaults to `1.0`.
- **max_delay** (float) - Optional - Maximum delay (in seconds) for backoff growth. Defaults to `30.0`.

### Request Example
```json
{
  "api_key": "YOUR_API_KEY",
  "organization": "YOUR_ORG_ID",
  "project": "YOUR_PROJECT_ID",
  "endpoint": "https://api.openai.com/v1/traces/ingest",
  "max_retries": 5,
  "base_delay": 2.0,
  "max_delay": 60.0
}
```

### Response
#### Success Response (200)
This endpoint does not return a success response body as it's used for ingesting trace data.

#### Response Example
(No specific response body for success)
```

--------------------------------

### Handoffs Multi-Agent System Pattern

Source: https://openai.github.io/openai-agents-python/agents

Illustrates the 'Handoffs' pattern for multi-agent systems, where an agent can delegate control to specialized sub-agents. The example shows a `triage_agent` that can hand off conversations to either a `booking_agent` or a `refund_agent` based on user intent.

```python
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, hand off to the booking agent. "
        "If they ask about refunds, hand off to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)

```

--------------------------------

### Get Text-to-Speech Model (Python)

Source: https://openai.github.io/openai-agents-python/ref/voice/models/openai_model_provider

Retrieves a Text-to-Speech (TTS) model from the OpenAI provider. If no model name is specified, it defaults to `DEFAULT_TTS_MODEL`. This function requires an initialized OpenAI client.

```python
def get_tts_model(self, model_name: str | None) -> TTSModel:
    """Get a text-to-speech model by name.

    Args:
        model_name: The name of the model to get.

    Returns:
        The text-to-speech model.
    """
    return OpenAITTSModel(model_name or DEFAULT_TTS_MODEL, self._get_client())
```

--------------------------------

### Get Model by Name using Python

Source: https://openai.github.io/openai-agents-python/zh/ref/models/interface

Defines an abstract method to retrieve a model by its name. This is a core function for model providers to make specific models accessible.

```python
import abc
from typing import Optional

# Assuming Model type is defined elsewhere
class Model:
    pass

class ABC(abc.ABC):
    pass

class ModelProvider(ABC):
    @abc.abstractmethod
    def get_model(self, model_name: Optional[str]) -> Model:
        """Get a model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The model.
        """
        pass

```

--------------------------------

### Get All Tools

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/agent

Retrieves all agent tools, including both MCP tools and function tools. This endpoint provides a comprehensive list of all tools accessible by the agent.

```APIDOC
## GET /api/tools/all

### Description
All agent tools, including MCP tools and function tools.

### Method
GET

### Endpoint
/api/tools/all

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
None

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of all available agent tools.

#### Response Example
```json
{
  "tools": [
    {
      "name": "example_mcp_tool",
      "description": "An example tool from MCP",
      "parameters": {},
      "return_direct": false
    },
    {
      "name": "example_function_tool",
      "description": "An example function tool",
      "parameters": {},
      "return_direct": false
    }
  ]
}
```
```

--------------------------------

### Define ComputerCreate Protocol for Tool Initialization in Python

Source: https://openai.github.io/openai-agents-python/ref/tool

A protocol defining the signature for a callable that initializes a computer for the current run context. It takes a RunContextWrapper and returns a ComputerT.

```python
from typing import Protocol, TypeVar, Any, MaybeAwaitable
from .run_context import RunContextWrapper

ComputerT_co = TypeVar("ComputerT_co", bound=Any, covariant=True)

class ComputerCreate(Protocol[ComputerT_co]):
    """Initializes a computer for the current run context."""

    def __call__(self, *, run_context: RunContextWrapper[Any]) -> MaybeAwaitable[ComputerT_co]: ...
```

--------------------------------

### Transcription Span API

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing/create

This function creates a new transcription span for speech-to-text processing. The span needs to be manually started and finished or used within a `with` statement.

```APIDOC
## POST /v1/audio/transcriptions (Conceptual)

### Description
Creates a new transcription span for speech-to-text processing. The span will not be started automatically; you should either use `with transcription_span() ...` or manually call `span.start()` and `span.finish()`.

### Method
POST (Conceptual - this is a Python function, not a direct HTTP endpoint)

### Endpoint
N/A (Python function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **model** (str | None) - Optional - The name of the model used for the speech-to-text.
- **input** (str | None) - Optional - The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes.
- **input_format** (str | None) - Optional - The format of the audio input. Defaults to 'pcm'.
- **output** (str | None) - Optional - The output of the speech-to-text transcription.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used.
- **disabled** (bool) - Optional - If True, the Span will not be recorded. Defaults to False.

### Request Example
```python
from agents.tracing import transcription_span

# Example usage within a 'with' statement
with transcription_span(model='whisper-1', input='base64_encoded_audio_data') as span:
    # Perform transcription tasks
    print(span.data.output)

# Example usage with manual start/finish
span = transcription_span(model='whisper-1', input='base64_encoded_audio_data')
span.start()
# Perform transcription tasks
span.finish()
```

### Response
#### Success Response (200)
- **Span[TranscriptionSpanData]** - The newly created speech-to-text span.

#### Response Example
```json
{
  "data": {
    "model": "whisper-1",
    "input": "base64_encoded_audio_data",
    "input_format": "pcm",
    "output": "This is a transcribed text.",
    "model_config": {}
  },
  "span_id": "generated_span_id",
  "parent_id": "optional_parent_id",
  "start_time": 1678886400.0,
  "end_time": 1678886405.0
}
```
```

--------------------------------

### Initialize SQLite Database Schema

Source: https://openai.github.io/openai-agents-python/ko/ref/memory/sqlite_session

Sets up the necessary database tables for storing session metadata and messages. This method creates `agent_sessions` and `agent_messages` tables if they do not already exist, ensuring the database schema is correctly configured for the session storage.

```python
    def _init_db_for_connection(self, conn: sqlite3.Connection) -> None:
        """Initialize the database schema for a specific connection."""
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

```

--------------------------------

### GET /items

Source: https://openai.github.io/openai-agents-python/ko/ref/extensions/memory/advanced_sqlite_session

Retrieves items from the current or a specified branch within an agent's memory session. It supports limiting the number of returned items and selecting a specific branch.

```APIDOC
## GET /items

### Description
Retrieves items from the current or specified branch. This method allows fetching a list of conversation items, optionally limiting the count and specifying a particular branch ID.

### Method
GET

### Endpoint
/items

### Parameters
#### Query Parameters
- **limit** (integer) - Optional - Maximum number of items to return. If not provided, all items are returned.
- **branch_id** (string) - Optional - The ID of the branch from which to retrieve items. If not provided, the current branch is used.

### Request Example
```json
{
  "limit": 10,
  "branch_id": "feature-branch-123"
}
```

### Response
#### Success Response (200)
- **items** (list) - A list of conversation items. Each item is a JSON object representing a message.

#### Response Example
```json
{
  "items": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    }
  ]
}
```

#### Error Response (e.g., 404 or 500)
- **error** (string) - A message describing the error.

```

--------------------------------

### Manage WebSocket Connection and Event Handling in Python

Source: https://openai.github.io/openai-agents-python/ko/ref/voice/models/openai_stt

This snippet demonstrates setting up and managing a WebSocket connection for real-time transcription. It includes handling connection setup, processing events, streaming audio, and managing listener tasks. Errors are caught and propagated. Dependencies include `asyncio` and potentially a custom `AgentsException`.

```python
                "wss://api.openai.com/v1/realtime?intent=transcription",
                additional_headers={
                    "Authorization": f"Bearer {self._client.api_key}",
                    "OpenAI-Log-Session": "1",
                },
            ) as ws:
                await self._setup_connection(ws)
                self._process_events_task = asyncio.create_task(self._handle_events())
                self._stream_audio_task = asyncio.create_task(self._stream_audio(self._input_queue))
                self.connected = True
                if self._listener_task:
                    await self._listener_task
                else:
                    logger.error("Listener task not initialized")
                    raise AgentsException("Listener task not initialized")
        except Exception as e:
            await self._output_queue.put(ErrorSentinel(e))
            raise e
```

--------------------------------

### transcription_span

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

Creates a new span for speech-to-text transcription operations. The span is not automatically started and requires manual start/finish or use within a 'with' statement.

```APIDOC
## POST /v1/spans/transcription

### Description
Creates a new span for speech-to-text transcription. This span can be managed manually (start/finish) or used with a context manager.

### Method
POST

### Endpoint
/v1/spans/transcription

### Parameters
#### Query Parameters
- **model** (str | None) - Optional - The name of the model used for the speech-to-text.
- **input** (str | None) - Optional - The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes.
- **input_format** (str | None) - Optional - The format of the audio input. Defaults to 'pcm'.
- **output** (str | None) - Optional - The output of the speech-to-text transcription.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span is used.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "model": "whisper-1",
  "input": "base64_encoded_audio_data",
  "input_format": "mp3",
  "model_config": {
    "temperature": 0.7
  }
}
```

### Response
#### Success Response (200)
- **span_id** (str) - The unique identifier for the created span.
- **status** (str) - The status of the span creation (e.g., 'created').

#### Response Example
```json
{
  "span_id": "span_abc123",
  "status": "created"
}
```
```

--------------------------------

### Get Agent Prompt in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/agent

This Python code snippet shows how to retrieve the prompt for an agent. It utilizes a `PromptUtil.to_model_input` method, which takes the agent's prompt configuration, the run context, and the agent instance as arguments. The function is asynchronous and returns a `ResponsePromptParam` or `None`.

```python
async def get_prompt(
        self,
        run_context: RunContextWrapper[TContext]
    ) -> ResponsePromptParam | None:
        """Get the prompt for the agent."""
        return await PromptUtil.to_model_input(self.prompt, run_context, self)
```

--------------------------------

### Prepare and Call OpenAI Chat Completions API (Python)

Source: https://openai.github.io/openai-agents-python/ref/models/openai_chatcompletions

This snippet prepares messages, tools, and other parameters, then calls the OpenAI Chat Completions API. It handles data conversion for tools and messages, conditional logging, and parameter mapping. It supports streaming and various model settings. Dependencies include `json`, `logging`, and OpenAI's client library.

```python
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)
        tools_param = converted_tools if converted_tools else omit

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            messages_json = json.dumps(
                converted_messages,
                indent=2,
                ensure_ascii=False,
            )
            tools_json = json.dumps(
                converted_tools,
                indent=2,
                ensure_ascii=False,
            )
            logger.debug(
                f"{messages_json}\n"
                f"Tools:\n{tools_json}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None
        store = ChatCmplHelpers.get_store_param(self._get_client(), model_settings)

        stream_options = ChatCmplHelpers.get_stream_options_param(
            self._get_client(), model_settings, stream=stream
        )

        stream_param: Literal[True] | Omit = True if stream else omit

        ret = await self._get_client().chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=tools_param,
            temperature=self._non_null_or_omit(model_settings.temperature),
            top_p=self._non_null_or_omit(model_settings.top_p),
            frequency_penalty=self._non_null_or_omit(model_settings.frequency_penalty),
            presence_penalty=self._non_null_or_omit(model_settings.presence_penalty),
            max_tokens=self._non_null_or_omit(model_settings.max_tokens),
            tool_choice=tool_choice,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
            stream=cast(Any, stream_param),
            stream_options=self._non_null_or_omit(stream_options),
            store=self._non_null_or_omit(store),
            reasoning_effort=self._non_null_or_omit(reasoning_effort),
            verbosity=self._non_null_or_omit(model_settings.verbosity),
            top_logprobs=self._non_null_or_omit(model_settings.top_logprobs),
            prompt_cache_retention=self._non_null_or_omit(model_settings.prompt_cache_retention),
            extra_headers=self._merge_headers(model_settings),
            extra_query=model_settings.extra_query,
            extra_body=model_settings.extra_body,
            metadata=self._non_null_or_omit(model_settings.metadata),
            **(model_settings.extra_args or {})
        )

        if isinstance(ret, ChatCompletion):
            return ret
```

--------------------------------

### Transcription Span API

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Creates a new span for speech-to-text transcription. The span needs to be manually started and finished or used within a 'with' statement.

```APIDOC
## transcription_span

### Description
Create a new transcription span. The span will not be started automatically, you should either do `with transcription_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
N/A (Function Signature)

### Endpoint
N/A (Function Signature)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
# Example usage within a 'with' statement
with transcription_span(model="whisper-1", input="base64_encoded_audio_data") as span:
    # Perform transcription or related operations
    pass

# Example usage with manual start/finish
span = transcription_span(model="whisper-1", input="base64_encoded_audio_data")
span.start()
# Perform transcription or related operations
span.finish()
```

### Response
#### Success Response (200)
- **Span[TranscriptionSpanData]** - The newly created speech-to-text span.

#### Response Example
```json
{
  "type": "Span[TranscriptionSpanData]",
  "description": "The newly created speech-to-text span."
}
```

### Parameters
- **model** (str | None) - Optional - The name of the model used for the speech-to-text. Defaults to `None`.
- **input** (str | None) - Optional - The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes. Defaults to `None`.
- **input_format** (str | None) - Optional - The format of the audio input. Defaults to `'pcm'`.
- **output** (str | None) - Optional - The output of the speech-to-text transcription. Defaults to `None`.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used. Defaults to `None`.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, a unique ID will be generated. Defaults to `None`.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used as the parent. Defaults to `None`.
- **disabled** (bool) - Optional - If True, the Span will be created but not recorded. Defaults to `False`.
```

--------------------------------

### List Available Tools

Source: https://openai.github.io/openai-agents-python/zh/ref/mcp/server

Retrieves a list of all tools currently available on the server. This is useful for discovering the capabilities of the agent.

```APIDOC
## GET /tools

### Description
List the tools available on the server.

### Method
GET

### Endpoint
/tools

### Parameters
#### Query Parameters
- **run_context** (RunContextWrapper[Any] | None) - Optional - The run context for the request.
- **agent** (AgentBase | None) - Optional - The agent to list tools for.

### Response
#### Success Response (200)
- **tools** (list[Tool]) - A list of available tools.

#### Response Example
```json
{
  "tools": [
    {
      "name": "example_tool",
      "description": "An example tool"
    }
  ]
}
```
```

--------------------------------

### Process WebSocket Connection Initiation

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_stt

This function initiates a WebSocket connection using websockets.connect. It is intended to be the starting point for establishing the connection to the server. It uses an async context manager for proper connection handling.

```python
async def _process_websocket_connection(self) -> None:
        try:
            async with websockets.connect(

```

--------------------------------

### Get Trace Name (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing/traces

Retrieves the human-readable name of a workflow trace. This name should be descriptive and meaningful, used for grouping and filtering traces in the dashboard.

```python
name: str

"""
Get the human-readable name of this workflow trace.

Returns:
    str: The workflow name (e.g., "Customer Service", "Data Processing")

Notes:
  * Should be descriptive and meaningful
  * Used for grouping and filtering in the dashboard
  * Helps identify the purpose of the trace
"""
```

--------------------------------

### Implement Input Guardrail with Agent

Source: https://openai.github.io/openai-agents-python/guardrails

This code defines an input guardrail that uses a separate agent to check if the user's input is a math homework question. It takes the input, runs it through the guardrail agent, and returns a GuardrailFunctionOutput indicating if a tripwire was triggered. The guardrail is then attached to a customer support agent.

```python
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_math_homework,
    )


agent = Agent(  
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[math_guardrail],
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except InputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped")

```

--------------------------------

### Initialize BackendSpanExporter in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/processors

Initializes the BackendSpanExporter with various configuration options including API key, organization, project, endpoint, and retry settings. It also sets up an httpx client for making requests. Dependencies include 'httpx' and 'os'.

```python
class BackendSpanExporter(TracingExporter):
    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        endpoint: str = "https://api.openai.com/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """
        Args:
            api_key: The API key for the "Authorization" header. Defaults to
                `os.environ["OPENAI_API_KEY"]` if not provided.
            organization: The OpenAI organization to use. Defaults to
                `os.environ["OPENAI_ORG_ID"]` if not provided.
            project: The OpenAI project to use. Defaults to
                `os.environ["OPENAI_PROJECT_ID"]` if not provided.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
        """
        self._api_key = api_key
        self._organization = organization
        self._project = project
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Keep a client open for connection pooling across multiple export calls
        self._client = httpx.Client(timeout=httpx.Timeout(timeout=60, connect=5.0))
```

--------------------------------

### GET /conversation_turns

Source: https://openai.github.io/openai-agents-python/zh/ref/extensions/memory/advanced_sqlite_session

Retrieves user messages from a specified conversation branch. This data is useful for browsing conversation history and making branching decisions.

```APIDOC
## GET /conversation_turns

### Description
Retrieves a list of user messages (turns) from a specific conversation branch. Each turn includes the message content, timestamp, and a flag indicating if branching is possible.

### Method
GET

### Endpoint
`/conversation_turns`

### Parameters
#### Query Parameters
- **branch_id** (string) - Optional - The ID of the branch to retrieve turns from. If not provided, the current branch is used.

### Request Example
```json
{
  "branch_id": "optional_branch_id_string"
}
```

### Response
#### Success Response (200)
- **turn** (integer) - The turn number in the conversation branch.
- **content** (string) - A truncated version of the user's message content.
- **full_content** (string) - The complete content of the user's message.
- **timestamp** (string) - The timestamp when the turn was created.
- **can_branch** (boolean) - Indicates if this turn can be branched (always true for user messages).

#### Response Example
```json
[
  {
    "turn": 1,
    "content": "Hello, how can I help you today? ...",
    "full_content": "Hello, how can I help you today?",
    "timestamp": "2023-10-27T10:00:00Z",
    "can_branch": true
  },
  {
    "turn": 2,
    "content": "I need assistance with my account. ...",
    "full_content": "I need assistance with my account.",
    "timestamp": "2023-10-27T10:05:00Z",
    "can_branch": true
  }
]
```
```

--------------------------------

### Python: Handle Span Start Event

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

This method is executed when a new span within a trace is initiated. It delegates the notification to each registered processor's `on_span_start` method, with built-in exception handling for robustness.

```python
def on_span_start(self, span: Span[Any]) -> None:
    """
    Called when a span is started.
    """
    for processor in self._processors:
        try:
            processor.on_span_start(span)
        except Exception as e:
            logger.error(f"Error in trace processor {processor} during on_span_start: {e}")
```

--------------------------------

### MCP Server Stdio Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Initializes a new MCP server using the stdio transport. This method allows for detailed configuration of the server's behavior, including process parameters, caching, timeouts, and retry mechanisms.

```APIDOC
## __init__ MCP Server Stdio

### Description
Create a new MCP server based on the stdio transport. This method allows for detailed configuration of the server's behavior, including process parameters, caching, timeouts, and retry mechanisms.

### Method
__init__

### Endpoint
N/A (Class constructor)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Parameters
- **params** (`MCPServerStdioParams`) - Required - The params that configure the server. This includes the command to run to start the server, the args to pass to the command, the environment variables to set for the server, the working directory to use when spawning the process, and the text encoding used when sending/receiving messages to the server.
- **cache_tools_list** (`bool`) - Optional - Whether to cache the tools list. If `True`, the tools list will be cached and only fetched from the server once. If `False`, the tools list will be fetched from the server on each call to `list_tools()`. The cache can be invalidated by calling `invalidate_tools_cache()`. You should set this to `True` if you know the server will not change its tools list, because it can drastically improve latency (by avoiding a round-trip to the server every time). Defaults to `False`.
- **name** (`str | None`) - Optional - A readable name for the server. If not provided, we'll create one from the command. Defaults to `None`.
- **client_session_timeout_seconds** (`float | None`) - Optional - The read timeout passed to the MCP ClientSession. Defaults to `5`.
- **tool_filter** (`ToolFilter`) - Optional - The tool filter to use for filtering tools. Defaults to `None`.
- **use_structured_content** (`bool`) - Optional - Whether to use `tool_result.structured_content` when calling an MCP tool. Defaults to `False` for backwards compatibility - most MCP servers still include the structured content in the `tool_result.content`, and using it by default will cause duplicate content. You can set this to `True` if you know the server will not duplicate the structured content in the `tool_result.content`. Defaults to `False`.
- **max_retry_attempts** (`int`) - Optional - Number of times to retry failed list_tools/call_tool calls. Defaults to `0`.
- **retry_backoff_seconds_base** (`float`) - Optional - The base delay, in seconds, for exponential backoff between retries. Defaults to `1.0`.
- **message_handler** (`MessageHandlerFnT | None`) - Optional - Optional handler invoked for session messages as delivered by the ClientSession. Defaults to `None`.

### Request Example
```json
{
  "params": { /* MCPServerStdioParams object */ },
  "cache_tools_list": false,
  "name": "my-agent-server",
  "client_session_timeout_seconds": 10.0,
  "tool_filter": null,
  "use_structured_content": true,
  "max_retry_attempts": 3,
  "retry_backoff_seconds_base": 2.0,
  "message_handler": null
}
```

### Response
#### Success Response (200)
N/A (This is a constructor, it does not return a response in the typical API sense. It initializes the object.)

#### Response Example
N/A
```

--------------------------------

### Get Default Model Settings

Source: https://openai.github.io/openai-agents-python/ref/models/default_models

Retrieves the model settings for the default model. It provides GPT-5 specific settings if the default model is GPT-5, otherwise legacy settings.

```APIDOC
## `get_default_model_settings`

### Description
Returns the default model settings. If the default model is a GPT-5 model, returns the GPT-5 default model settings. Otherwise, returns the legacy default model settings.

### Method
N/A (Function)

### Endpoint
N/A (Function)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
get_default_model_settings()
```

### Response
#### Success Response (ModelSettings)
- **settings** (ModelSettings) - An object containing the default model settings.

#### Response Example
```json
{
  "settings": {
    "temperature": 0.7,
    "max_tokens": 1024
    // ... other model settings
  }
}
```
```

--------------------------------

### Create SQLAlchemy Session from URL (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/extensions/memory/sqlalchemy_session

The `from_url` class method instantiates a SQLAlchemySession using a provided database URL. It accepts session ID, the URL string, and optional engine keyword arguments for SQLAlchemy's create_async_engine. It returns a configured SQLAlchemySession instance.

```python
from sqlalchemy.ext.asyncio import create_async_engine

# Assuming SQLAlchemySession and TResponseInputItem are defined elsewhere
# class SQLAlchemySession:
#     def __init__(self, session_id: str, engine, **kwargs):
#         self.session_id = session_id
#         self._engine = engine
#         # ... other initializations ...

#     @classmethod
#     def from_url(
#         cls,
#         session_id: str,
#         *, 
#         url: str,
#         engine_kwargs: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> SQLAlchemySession:
#         """Create a session from a database URL string."""
#         engine_kwargs = engine_kwargs or {}
#         engine = create_async_engine(url, **engine_kwargs)
#         return cls(session_id, engine=engine, **kwargs)

# Example Usage:
# session = SQLAlchemySession.from_url(
#     session_id="my_conversation_id",
#     url="postgresql+asyncpg://user:pass@host/db"
# )
```

--------------------------------

### Get Speech-to-Text Model (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_provider

Retrieves a speech-to-text (STT) model by its name. If no model name is provided, it defaults to a predefined model. This method requires an initialized OpenAI client to function.

```python
def get_stt_model(self, model_name: str | None) -> STTModel:
    """Get a speech-to-text model by name.

    Args:
        model_name: The name of the model to get.

    Returns:
        The speech-to-text model.
    """
    return OpenAISTTModel(model_name or DEFAULT_STT_MODEL, self._get_client())

```

--------------------------------

### Get All Function Tools from MCP Servers (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/util

Retrieves all function tools from a list of MCP servers. It checks for duplicate tool names across servers and raises a UserError if any are found. Dependencies include MCPServer, Tool, RunContextWrapper, AgentBase, and UserError.

```python
class MCPUtil:
    @classmethod
    async def get_all_function_tools(
        cls,
        servers: list["MCPServer"],
        convert_schemas_to_strict: bool,
        run_context: RunContextWrapper[Any],
        agent: "AgentBase",
    ) -> list[Tool]:
        """Get all function tools from a list of MCP servers."""
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(
                server, convert_schemas_to_strict, run_context, agent
            )
            server_tool_names = {tool.name for tool in server_tools}
            if len(server_tool_names & tool_names) > 0:
                raise UserError(
                    f"Duplicate tool names found across MCP servers: "
                    f"{server_tool_names & tool_names}"
                )
            tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools
```

--------------------------------

### on_trace_start

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/processor_interface

This method is called synchronously when a new trace begins execution. It should return quickly to avoid blocking the execution. Any errors encountered should be handled internally.

```APIDOC
## on_trace_start

### Description
Called when a new trace begins execution.

### Method
`on_trace_start`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **trace** (`Trace`) - Required - The trace that started. Contains workflow name and metadata.

### Request Example
```json
{
  "trace": { "workflow_name": "example_workflow", "metadata": {} }
}
```

### Response
#### Success Response (200)
None

#### Response Example
```json
null
```

### Notes
- Called synchronously on trace start
- Should return quickly to avoid blocking execution
- Any errors should be caught and handled internally
```

--------------------------------

### Set Global Trace Provider - Python

Source: https://openai.github.io/openai-agents-python/ref/tracing/setup

Configures the global trace provider used by tracing utilities. This function takes a `TraceProvider` object as input and sets it as the global provider. It is essential for initializing the tracing mechanism.

```python
def set_trace_provider(provider: TraceProvider) -> None:
    """Set the global trace provider used by tracing utilities."""
    global GLOBAL_TRACE_PROVIDER
    GLOBAL_TRACE_PROVIDER = provider
```

--------------------------------

### Get All Mappings from MultiProviderMap (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/models/multi_provider

Implements the `get_mapping` method for the MultiProviderMap class. This method returns a shallow copy of the internal dictionary that stores the prefix to ModelProvider mappings. It takes no arguments and returns a dictionary.

```python
def get_mapping(self) -> dict[str, ModelProvider]:
    """Returns a copy of the current prefix -> ModelProvider mapping."""
    return self._mapping.copy()

```

--------------------------------

### Create Trace

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Creates a new trace object. This trace is not automatically started and requires manual management using context managers or explicit start/finish calls. It allows for associating metadata and grouping identifiers with the trace.

```APIDOC
## POST /trace

### Description
Creates a new trace object. This trace is not automatically started and requires manual management using context managers or explicit start/finish calls. It allows for associating metadata and grouping identifiers with the trace.

### Method
POST

### Endpoint
/trace

### Parameters
#### Query Parameters
- **workflow_name** (str) - Required - The name of the logical app or workflow. For example, you might provide "code_bot" for a coding agent, or "customer_support_agent" for a customer support agent.
- **trace_id** (str | None) - Optional - The ID of the trace. If not provided, an ID will be generated. It is recommended to use `util.gen_trace_id()` for correctly formatted IDs.
- **group_id** (str | None) - Optional - An identifier to link multiple traces from the same conversation or process, such as a chat thread ID.
- **metadata** (dict[str, Any] | None) - Optional - A dictionary of additional user-defined information to attach to the trace.
- **disabled** (bool) - Optional - If True, the trace object is returned but not recorded. Defaults to `False`.

### Request Example
```json
{
  "workflow_name": "customer_support_agent",
  "trace_id": "optional_trace_123",
  "group_id": "chat_thread_abc",
  "metadata": {
    "user_query": "How do I reset my password?",
    "priority": "high"
  },
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Trace** (Trace) - The newly created trace object.

#### Response Example
```json
{
  "trace_id": "generated_or_provided_trace_id",
  "workflow_name": "customer_support_agent",
  "status": "created"
}
```
```

--------------------------------

### Get Playback State in Python

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/openai_realtime

Retrieves the current playback state, prioritizing data from a playback tracker. If unavailable, it attempts to derive the state from audio state tracking, calculating elapsed time since the last audio item was received.

```python
def _get_playback_state(self) -> RealtimePlaybackState:
    if self._playback_tracker:
        return self._playback_tracker.get_state()

    if last_audio_item_id := self._audio_state_tracker.get_last_audio_item():
        item_id, item_content_index = last_audio_item_id
        audio_state = self._audio_state_tracker.get_state(item_id, item_content_index)
        if audio_state:
            elapsed_ms = (
                datetime.now() - audio_state.initial_received_time
            ).total_seconds() * 1000
            return {
                "current_item_id": item_id,
                "current_item_content_index": item_content_index,
                "elapsed_ms": elapsed_ms,
            }

    return {
        "current_item_id": None,
        "current_item_content_index": None,
        "elapsed_ms": None,
    }
```

--------------------------------

### Create Transcription Span

Source: https://openai.github.io/openai-agents-python/ref/tracing/create

This endpoint allows you to create a new span for speech-to-text transcription. The span is not started automatically and requires manual start/finish calls or use within a `with` statement.

```APIDOC
## POST /transcription_span

### Description
Creates a new transcription span for speech-to-text processing. The span needs to be explicitly started and finished, or managed using a `with` statement.

### Method
POST

### Endpoint
/transcription_span

### Parameters
#### Query Parameters
- **model** (str | None) - Optional - The name of the model used for the speech-to-text.
- **input** (str | None) - Optional - The audio input of the speech-to-text transcription, as a base64 encoded string of audio bytes.
- **input_format** (str | None) - Optional - The format of the audio input (defaults to "pcm").
- **output** (str | None) - Optional - The output of the speech-to-text transcription.
- **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "model": "whisper-1",
  "input": "BASE64_ENCODED_AUDIO_DATA",
  "input_format": "mp3",
  "model_config": {
    "temperature": 0.2
  }
}
```

### Response
#### Success Response (200)
- **Span[TranscriptionSpanData]** - The newly created speech-to-text span.
```

--------------------------------

### Integrate Realtime Agents with SIP Calls in Python

Source: https://openai.github.io/openai-agents-python/realtime/guide

This Python code shows how to integrate realtime agents with phone calls using the Realtime Calls API and `OpenAIRealtimeSIPModel`. It sets up a `RealtimeRunner` with the SIP model and configures call-specific settings like turn detection. The session handles media negotiation over SIP and automatically closes when the caller hangs up.

```python
from agents.realtime import RealtimeAgent, RealtimeRunner
from agents.realtime.openai_realtime import OpenAIRealtimeSIPModel

# Assuming 'agent' is a pre-configured RealtimeAgent instance
agent = RealtimeAgent(name="Assistant", instructions="...") # Example agent

runner = RealtimeRunner(
    starting_agent=agent,
    model=OpenAIRealtimeSIPModel(),
)

# call_id_from_webhook would be received from an incoming call webhook
call_id_from_webhook = "your_call_id_here"

async with await runner.run(
    model_config={
        "call_id": call_id_from_webhook,
        "initial_model_settings": {
            "turn_detection": {"type": "semantic_vad", "interrupt_response": True},
        },
    },
) as session:
    async for event in session:
        # Handle session events
        pass

```

--------------------------------

### MCPServerStdio Class Documentation

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Provides detailed documentation for the MCPServerStdio class, including its constructor and properties.

```APIDOC
## MCPServerStdio Class

### Description

MCP server implementation that uses the stdio transport. See the [spec](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for details.

### Constructor

```python
def __init__(
    self,
    params: MCPServerStdioParams,
    cache_tools_list: bool = False,
    name: str | None = None,
    client_session_timeout_seconds: float | None = 5,
    tool_filter: ToolFilter = None,
    use_structured_content: bool = False,
    max_retry_attempts: int = 0,
    retry_backoff_seconds_base: float = 1.0,
    message_handler: MessageHandlerFnT | None = None,
):
    """Create a new MCP server based on the stdio transport.

    Args:
        params: The params that configure the server. This includes the command to run to
            start the server, the args to pass to the command, the environment variables to
            set for the server, the working directory to use when spawning the process, and
            the text encoding used when sending/receiving messages to the server.
        cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
            cached and only fetched from the server once. If `False`, the tools list will be
            fetched from the server on each call to `list_tools()`. The cache can be
            invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
            if you know the server will not change its tools list, because it can drastically
            improve latency (by avoiding a round-trip to the server every time).
        name: A readable name for the server. If not provided, we'll create one from the
            command.
        client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
        tool_filter: The tool filter to use for filtering tools.
        use_structured_content: Whether to use `tool_result.structured_content` when calling an
            MCP tool. Defaults to False for backwards compatibility - most MCP servers still
            include the structured content in the `tool_result.content`, and using it by
            default will cause duplicate content. You can set this to True if you know the
            server will not duplicate the structured content in the `tool_result.content`.
        max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
            Defaults to no retries.
        retry_backoff_seconds_base: The base delay, in seconds, for exponential
            backoff between retries.
        message_handler: Optional handler invoked for session messages as delivered by the
            ClientSession.
    """
```

### Methods

#### `create_streams()`

```python
def create_streams(
    self,
) -> AbstractAsyncContextManager[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        GetSessionIdCallback | None,
    ]
]:
    """Create the streams for the server."""
```

### Properties

#### `name`

```python
@property
name: str
```

A readable name for the server.

```

--------------------------------

### Using Usage Metrics in Run Hooks

Source: https://openai.github.io/openai-agents-python/usage

Provides an example of how to access and utilize token usage data within `RunHooks`. The `context` object passed to hook methods like `on_agent_end` includes the `usage` attribute, allowing logging or other actions based on usage metrics at key points in the agent's lifecycle.

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

--------------------------------

### Response Span Creation

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Creates a new span for tracking responses. Similar to `mcp_tools_span`, this span requires manual start and finish or use with a `with` statement. It is designed to capture OpenAI Response objects.

```APIDOC
## POST /v1/tracing/response_span

### Description
Create a new response span. The span will not be started automatically; you should either use `with response_span() ...` or call `span.start()` and `span.finish()` manually.

### Method
POST

### Endpoint
`/v1/tracing/response_span`

### Parameters
#### Request Body
- **response** (Response | None) - Optional - The OpenAI Response object.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated. It is recommended to use `util.gen_span_id()` for correctly formatted IDs.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used as the parent.
- **disabled** (bool) - Optional - If True, a Span will be returned but not recorded. Defaults to False.

### Request Example
```json
{
  "response": {"id": "resp_abc123", "object": "response"},
  "span_id": "generated_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span** (Span[ResponseSpanData]) - The created response span object.

#### Response Example
```json
{
  "span": "<span_object_representation>"
}
```
```

--------------------------------

### Span Lifecycle Methods (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

Provides the abstract methods for managing the lifecycle of a span. The `start` method initializes the span, optionally marking it as the current span. The `finish` method completes the span, also with an option to reset the current span.

```python
@abc.abstractmethod
def start(self, mark_as_current: bool = False):
    """
    Start the span.

    Args:
        mark_as_current: If true, the span will be marked as the current span.
    """
    pass

```

```python
@abc.abstractmethod
def finish(self, reset_current: bool = False) -> None:
    """
    Finish the span.

    Args:
        reset_current: If true, the span will be reset as the current span.
    """
    pass

```

--------------------------------

### Initialize MCPServerStreamableHttp with Streamable HTTP

Source: https://openai.github.io/openai-agents-python/zh/ref/mcp/server

Initializes the MCPServerStreamableHttp class, setting up an MCP server with Streamable HTTP transport. It accepts parameters for server configuration, caching, timeouts, tool filtering, and retry logic. The constructor ensures proper inheritance from the base class and stores the provided parameters.

```python
class MCPServerStreamableHttp(_MCPServerWithClientSession):
    """MCP server implementation that uses the Streamable HTTP transport. See the [spec]
    (https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http)
    for details.
    """

    def __init__(
        self,
        params: MCPServerStreamableHttpParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new MCP server based on the Streamable HTTP transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, the
                timeout for the Streamable HTTP connection, whether we need to
                terminate on close, and an optional custom HTTP client factory.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
        )

        self.params = params
        self._name = name or f"streamable_http: {self.params['url']}"
```

```python
def __init__(
    self,
    params: MCPServerStreamableHttpParams,
    cache_tools_list: bool = False,
    name: str | None = None,
    client_session_timeout_seconds: float | None = 5,
    tool_filter: ToolFilter = None,
    use_structured_content: bool = False,
    max_retry_attempts: int = 0,
    retry_backoff_seconds_base: float = 1.0,
    message_handler: MessageHandlerFnT | None = None,
):
    super().__init__(
        cache_tools_list,
        client_session_timeout_seconds,
        tool_filter,
        use_structured_content,
        max_retry_attempts,
        retry_backoff_seconds_base,
        message_handler=message_handler,
    )

    self.params = params
    self._name = name or f"streamable_http: {self.params['url']}"
```

--------------------------------

### POST /connect

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/model

Establishes a connection to the realtime model. This method should be called before sending any events or adding listeners.

```APIDOC
## POST /connect

### Description
Establish a connection to the model and keep it alive.

### Method
POST

### Endpoint
/connect

#### Parameters

##### Request Body
- **options** (RealtimeModelConfig) - Required - Configuration options for the realtime model connection.
```

--------------------------------

### Span Class

Source: https://openai.github.io/openai-agents-python/ref/tracing/spans

The base class for representing traceable operations. It defines methods for starting, finishing, setting errors, and accessing span data within a trace.

```APIDOC
## Span Class

### Description
Represents a single operation within a trace, tracking timing, relationships, and operation-specific data. Spans can be used with context managers for reliable start/finish and include mechanisms for error handling.

### Usage Examples
```python
# Creating a custom span with data
with custom_span("database_query", {
    "operation": "SELECT",
    "table": "users"
}) as span:
    results = await db.query("SELECT * FROM users")
    span.set_output({"count": len(results)})

# Handling errors within a span
with custom_span("risky_operation") as span:
    try:
        result = perform_risky_operation()
    except Exception as e:
        span.set_error({
            "message": str(e),
            "data": {"operation": "risky_operation"}
        })
        raise
```

### Notes
- Spans automatically nest under the current trace.
- Use context managers (`with` statement) for reliable span start and finish.
- Include relevant data in spans but avoid sensitive information.
- Handle errors properly using the `set_error()` method.

### Abstract Methods
- `trace_id`: Property returning the ID of the trace this span belongs to.
- `span_id`: Property returning the unique identifier for this span.
- `span_data`: Property returning operation-specific data for this span.
- `start(mark_as_current: bool = False)`: Starts the span. `mark_as_current` makes it the current span.
- `finish(reset_current: bool = False)`: Finishes the span. `reset_current` resets the current span.
- `__enter__()`: Enters the runtime context related to this object.
- `__exit__(exc_type, exc_val, exc_tb)`: Exits the runtime context, handling exceptions.
- `parent_id`: Property returning the ID of the parent span, or `None` if it's a root span.
- `set_error(error: SpanError)`: Sets error information for the span.
- `error`: Property returning error details if an error occurred, `None` otherwise.
- `export()`: Exports the span data as a dictionary.
- `started_at`: Property returning the ISO format timestamp of when the span started, or `None`.
- `ended_at`: Property returning the ISO format timestamp of when the span ended, or `None`.
```

--------------------------------

### Configure OpenAI Session Request

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/openai_realtime

Constructs a session creation request for OpenAI's Realtime API, including model settings, session ID, and tool configurations. It handles optional parameters like max output tokens and tool choice. Dependencies include `Any`, `cast`, `Tool`, `Handoff`, `UserError`, `FunctionTool`, and `OpenAISessionFunction`.

```python
def _session_create_request(self, _passed_prompt: dict, variables: list[str], model_settings: dict) -> SessionCreateRequest:
    session_create_request = SessionCreateRequest(
        id=_passed_prompt["id"],
        variables=variables,
        version=_passed_prompt.get("version"),
    )

    if "max_output_tokens" in model_settings:
        session_create_request.max_output_tokens = cast(
            Any, model_settings.get("max_output_tokens")
        )

    if "tool_choice" in model_settings:
        session_create_request.tool_choice = cast(Any, model_settings.get("tool_choice"))

    return session_create_request

def _tools_to_session_tools(self, tools: list[Tool], handoffs: list[Handoff]) -> list[OpenAISessionFunction]:
    converted_tools: list[OpenAISessionFunction] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            raise UserError(f"Tool {tool.name} is unsupported. Must be a function tool.")
        converted_tools.append(
            OpenAISessionFunction(
                name=tool.name,
                description=tool.description,
                parameters=tool.params_json_schema,
                type="function",
            )
        )

    for handoff in handoffs:
        converted_tools.append(
            OpenAISessionFunction(
                name=handoff.tool_name,
                description=handoff.tool_description,
                parameters=handoff.input_json_schema,
                type="function",
            )
        )

    return converted_tools
```

--------------------------------

### Connect to MCP Server Asynchronously in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

The `connect` method establishes an asynchronous connection to the MCP server. It initializes communication streams, creates a client session, and attempts to initialize the server. Error handling is included to log issues and initiate cleanup if connection fails. This function is essential for starting server interactions.

```python
async def connect(self):
    """Connect to the server."""
    try:
        transport = await self.exit_stack.enter_async_context(self.create_streams())
        # streamablehttp_client returns (read, write, get_session_id)
        # sse_client returns (read, write)

        read, write, *_ = transport

        session = await self.exit_stack.enter_async_context(
            ClientSession(
                read,
                write,
                timedelta(seconds=self.client_session_timeout_seconds)
                if self.client_session_timeout_seconds
                else None,
                message_handler=self.message_handler,
            )
        )
        server_result = await session.initialize()
        self.server_initialize_result = server_result
        self.session = session
    except Exception as e:
        logger.error(f"Error initializing MCP server: {e}")
        await self.cleanup()
        raise
```

--------------------------------

### Convert OpenAI Agent Tools to Dictionary Representation (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

Converts various OpenAI agent tool instances into a dictionary format suitable for serialization. It handles specific tool types like FileSearchTool, ComputerTool, HostedMCPTool, ApplyPatchTool, ShellTool, ImageGenerationTool, CodeInterpreterTool, and LocalShellTool, extracting relevant configuration and parameters. Raises UserError for unknown tool types or improperly initialized Computer tools.

```python
def _convert_tool(self, tool: Tool):
    if isinstance(tool, SearchTool):
        converted_tool = {
            "type": "search",
            "user_location": tool.user_location,
            "search_context_size": tool.search_context_size,
        }
        includes = None
    elif isinstance(tool, FileSearchTool):
        converted_tool = {
            "type": "file_search",
            "vector_store_ids": tool.vector_store_ids,
        }
        if tool.max_num_results:
            converted_tool["max_num_results"] = tool.max_num_results
        if tool.ranking_options:
            converted_tool["ranking_options"] = tool.ranking_options
        if tool.filters:
            converted_tool["filters"] = tool.filters

        includes = "file_search_call.results" if tool.include_search_results else None
    elif isinstance(tool, ComputerTool):
        computer = tool.computer
        if not isinstance(computer, (Computer, AsyncComputer)):
            raise UserError(
                "Computer tool is not initialized for serialization. Call "
                "resolve_computer({ tool, run_context }) with a run context first "
                "when building payloads manually."
            )
        converted_tool = {
            "type": "computer_use_preview",
            "environment": computer.environment,
            "display_width": computer.dimensions[0],
            "display_height": computer.dimensions[1],
        }
        includes = None
    elif isinstance(tool, HostedMCPTool):
        converted_tool = tool.tool_config
        includes = None
    elif isinstance(tool, ApplyPatchTool):
        converted_tool = cast(ToolParam, {"type": "apply_patch"})
        includes = None
    elif isinstance(tool, ShellTool):
        converted_tool = cast(ToolParam, {"type": "shell"})
        includes = None
    elif isinstance(tool, ImageGenerationTool):
        converted_tool = tool.tool_config
        includes = None
    elif isinstance(tool, CodeInterpreterTool):
        converted_tool = tool.tool_config
        includes = None
    elif isinstance(tool, LocalShellTool):
        converted_tool = {
            "type": "local_shell",
        }
        includes = None
    else:
        raise UserError(f"Unknown tool type: {type(tool)}, tool")

    return converted_tool, includes
```

--------------------------------

### Response Span Creation

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/create

Provides functionality to create a new response span for tracing purposes. The span requires manual starting and finishing or can be used within a `with` statement.

```APIDOC
## POST /agents/tracing/response_span

### Description
Creates a new response span to track OpenAI responses. The span needs to be explicitly started and finished, or managed using a context manager.

### Method
POST

### Endpoint
`/agents/tracing/response_span`

### Parameters
#### Query Parameters
- **response** (Response | None) - Optional - The OpenAI Response object.
- **span_id** (str | None) - Optional - The ID for the span. If not provided, one will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. Defaults to the current trace/span.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "response": { /* OpenAI Response object */ },
  "span_id": "optional-span-id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[ResponseSpanData]** - The newly created response span.

#### Response Example
```json
{
  "span_id": "generated-or-provided-span-id",
  "status": "running" /* or "finished" */
}
```
```

--------------------------------

### Python: Handle Trace Start Event

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

This function is called when a new trace begins. It iterates through registered processors and calls their respective `on_trace_start` method. It includes error handling to log issues encountered during processor execution.

```python
def on_trace_start(self, trace: Trace) -> None:
    """
    Called when a trace is started.
    """
    for processor in self._processors:
        try:
            processor.on_trace_start(trace)
        except Exception as e:
            logger.error(f"Error in trace processor {processor} during on_trace_start: {e}")
```

--------------------------------

### GET /conversations/tool_usage

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Retrieves a list of all tool usage events, categorized by tool name, usage count, and the turn number in which they occurred, for a specified branch. Defaults to the current branch if no branch ID is provided.

```APIDOC
## GET /conversations/tool_usage

### Description
Retrieves a list of all tool usage events, categorized by tool name, usage count, and the turn number in which they occurred, for a specified branch. This endpoint is useful for analyzing tool adoption and patterns within a conversation.

### Method
GET

### Endpoint
`/conversations/tool_usage`

### Parameters
#### Query Parameters
- **branch_id** (string) - Optional - The ID of the branch for which to retrieve tool usage. If not provided, the current branch is used.

### Request Example
```json
{
  "branch_id": "optional_branch_id"
}
```

### Response
#### Success Response (200)
- **tool_name** (string) - The name of the tool that was used.
- **usage_count** (integer) - The number of times the tool was used.
- **turn_number** (integer) - The turn number in which the tool usage occurred.

#### Response Example
```json
[
  {
    "tool_name": "python",
    "usage_count": 2,
    "turn_number": 1
  },
  {
    "tool_name": "search",
    "usage_count": 1,
    "turn_number": 2
  }
]
```
```

--------------------------------

### POST /agents/run

Source: https://openai.github.io/openai-agents-python/ja/ref/run

Initiates an agent workflow from a specified starting agent. The agent will iterate through a process of receiving input, generating output, and potentially calling tools or handing off to other agents until a final output is produced or an exception is raised.

```APIDOC
## POST /agents/run

### Description
Runs a workflow starting at the given agent. The agent will run in a loop until a final output is generated. The loop runs like so:
1. The agent is invoked with the given input.
2. If there is a final output (i.e. the agent produces something of type `agent.output_type`), the loop terminates.
3. If there's a handoff, we run the loop again, with the new agent.
4. Else, we run tool calls (if any), and re-run the loop.

In two cases, the agent may raise an exception:
1. If the `max_turns` is exceeded, a `MaxTurnsExceeded` exception is raised.
2. If a guardrail tripwire is triggered, a `GuardrailTripwireTriggered` exception is raised.

**Note:** Only the first agent's input guardrails are run.

### Method
POST

### Endpoint
/agents/run

### Parameters
#### Request Body
- **starting_agent** (`Agent[TContext]`) - Required - The starting agent to run.
- **input** (`str | list[TResponseInputItem]`) - Required - The initial input to the agent. You can pass a single string for a user message, or a list of input items.
- **context** (`TContext | None`) - Optional - The context to run the agent with. Defaults to `None`.
- **max_turns** (`int`) - Optional - The maximum number of turns to run the agent for. A turn is defined as one AI invocation (including any tool calls that might occur). Defaults to `DEFAULT_MAX_TURNS`.
- **hooks** (`RunHooks[TContext] | None`) - Optional - An object that receives callbacks on various lifecycle events. Defaults to `None`.
- **run_config** (`RunConfig | None`) - Optional - Global settings for the entire agent run. Defaults to `None`.
- **previous_response_id** (`str | None`) - Optional - The ID of the previous response. If using OpenAI models via the Responses API, this allows you to skip passing in input from the previous turn. Defaults to `None`.
- **auto_previous_response_id** (`bool`) - Optional - Whether to automatically set the `previous_response_id`. Defaults to `False`.
- **conversation_id** (`str | None`) - Optional - The conversation ID. If provided, the conversation will be used to read and write items. Every agent will have access to the conversation history so far, and its output items will be written to the conversation. We recommend only using this if you are exclusively using OpenAI models. Defaults to `None`.
- **session** (`Session | None`) - Optional - A session for automatic conversation history management. Defaults to `None`.

### Response
#### Success Response (200)
- **RunResult** (`RunResult`) - A run result containing all the inputs, guardrail results, and the output of the last agent. Agents may perform handoffs, so we don't know the specific type of the output.
```

--------------------------------

### MCPServer __init__ Method (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Initializes an MCPServer instance. It accepts an optional boolean argument `use_structured_content` to control whether the `tool_result.structured_content` is used when calling an MCP tool. The default is False for backward compatibility.

```python
def __init__(self, use_structured_content: bool = False):
    """
    Args:
        use_structured_content: Whether to use `tool_result.structured_content` when calling an
            MCP tool.Defaults to False for backwards compatibility - most MCP servers still
            include the structured content in the `tool_result.content`, and using it by
            default will cause duplicate content. You can set this to True if you know the
            server will not duplicate the structured content in the `tool_result.content`.
    """
    self.use_structured_content = use_structured_content

```

--------------------------------

### SQLAlchemySession: Get Items

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/sqlalchemy_session

Retrieves conversation history items from the database for a given session. It supports retrieving all items or a specified number of the latest items, ordered chronologically.

```python
async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
    """Retrieve the conversation history for this session.

    Args:
        limit: Maximum number of items to retrieve. If None, retrieves all items.
               When specified, returns the latest N items in chronological order.

    Returns:
        List of input items representing the conversation history
    """
    await self._ensure_tables()
    async with self._session_factory() as sess:
        if limit is None:
            stmt = (
                select(self._messages.c.message_data)
                .where(self._messages.c.session_id == self.session_id)
                .order_by(
                    self._messages.c.created_at.asc(),
                    self._messages.c.id.asc(),
                )
            )
        else:
            stmt = (
                select(self._messages.c.message_data)
                .where(self._messages.c.session_id == self.session_id)
                # Use DESC + LIMIT to get the latest N
                # then reverse later for chronological order.
                .order_by(
                    self._messages.c.created_at.desc(),
                    self._messages.c.id.desc(),
                )
                .limit(limit)
            )

        result = await sess.execute(stmt)
        rows: list[str] = [row[0] for row in result.all()]

        if limit is not None:
            rows.reverse()

        items: list[TResponseInputItem] = []
        for raw in rows:
            try:
                items.append(await self._deserialize_item(raw))
            except json.JSONDecodeError:
                # Skip corrupted rows
                continue
        return items
```

--------------------------------

### Function Span Creation

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

This section describes the `function_span` function used to create a new span for a function. The span is not started automatically and requires manual start/finish or usage within a `with` statement.

```APIDOC
## Function Span Creation

### Description
Creates a new span for a function call. This span needs to be manually started and finished or managed using a `with` statement.

### Method
`function_span`

### Endpoint
N/A (This is a function within the Python library, not a REST endpoint)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None (This is a function call with arguments)

**Arguments:**
- **name** (`str`) - Required - The name of the function.
- **input** (`str | None`) - Optional - The input provided to the function. Defaults to `None`.
- **output** (`str | None`) - Optional - The output returned by the function. Defaults to `None`.
- **span_id** (`str | None`) - Optional - The unique identifier for the span. If not provided, an ID will be generated. Defaults to `None`.
- **parent** (`Trace | Span[Any] | None`) - Optional - The parent trace or span. If not provided, the current trace/span is used as the parent. Defaults to `None`.
- **disabled** (`bool`) - Optional - If set to `True`, the span will not be recorded. Defaults to `False`.

### Request Example
```python
from agents.tracing import function_span

# Example using a with statement
with function_span(name="my_function", input="some_input") as span:
    # Function logic here
    span.output = "some_output"

# Example with manual start and finish
span = function_span(name="another_function")
span.start()
# Function logic here
span.output = "another_output"
span.finish()
```

### Response
#### Success Response
- **Span[FunctionSpanData]** - The newly created function span object, which can be used to track the function's execution details.
```

--------------------------------

### speech_span

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Creates a new speech span for tracing text-to-speech operations. The span can be managed manually using `start()` and `finish()` or with a `with` statement. It allows detailed configuration of the model, input/output, and span properties.

```APIDOC
## speech_span

### Description
Creates a new speech span. The span will not be started automatically; you should either use `with speech_span() ...` or call `span.start()` and `span.finish()` manually.

### Method
N/A (This is a Python function, not an HTTP endpoint)

### Endpoint
N/A

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **model** (str | None) - Optional - The name of the model used for the text-to-speech.
* **input** (str | None) - Optional - The text input of the text-to-speech.
* **output** (str | None) - Optional - The audio output of the text-to-speech as base64 encoded string of PCM audio bytes.
* **output_format** (str | None) - Optional - The format of the audio output (defaults to "pcm").
* **model_config** (Mapping[str, Any] | None) - Optional - The model configuration (hyperparameters) used.
* **first_content_at** (str | None) - Optional - The time of the first byte of the audio output.
* **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated. It is recommended to use `util.gen_span_id()` for correctly formatted IDs.
* **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used as the parent.
* **disabled** (bool) - Optional - If True, a Span will be returned but not recorded. Defaults to False.

### Request Example
```python
# Example usage with a 'with' statement
with speech_span(model="tts-1", input="Hello, world!") as span:
    # Perform speech synthesis operations
    print(f"Speech span created with ID: {span.id}")

# Example manual start and finish
span = speech_span(model="tts-1", input="Another test.")
try:
    span.start()
    # Perform speech synthesis operations
finally:
    span.finish()
```

### Response
#### Success Response (Span[SpeechSpanData])
Returns a `Span` object containing `SpeechSpanData`.

#### Response Example
```json
{
  "id": "some-span-id",
  "name": "speech",
  "start_time": "2023-10-27T10:00:00Z",
  "end_time": "2023-10-27T10:00:05Z",
  "status": "ok",
  "data": {
    "model": "tts-1",
    "input": "Hello, world!",
    "output": "base64encodedpcmdata...",
    "output_format": "pcm",
    "model_config": {},
    "first_content_at": "2023-10-27T10:00:01Z"
  }
}
```
```

--------------------------------

### Set Up Streamable HTTP MCP Server

Source: https://openai.github.io/openai-agents-python/mcp

This code sets up a streamable HTTP MCP server for managing network connections directly. It's suitable for scenarios where you control the transport or need to run the server within your infrastructure with low latency. The server is configured with connection parameters, retry attempts, and tool caching.

```python
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings

async def main() -> None:
    token = os.environ["MCP_SERVER_TOKEN"]
    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    ) as server:
        agent = Agent(
            name="Assistant",
            instructions="Use the MCP tools to answer the questions.",
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="required"),
        )

        result = await Runner.run(agent, "Add 7 and 22.")
        print(result.final_output)

asyncio.run(main())

```

--------------------------------

### Get Default Batch Trace Processor (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/processors

Retrieves the default batch trace processor. This processor handles the batching and exporting of traces and spans to the backend. It is a globally configured instance.

```python
def default_processor() -> BatchTraceProcessor:
    """The default processor, which exports traces and spans to the backend in batches."""
    return _global_processor

```

--------------------------------

### Configure Stream Options and Extra Arguments for Model Calls in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/litellm

Sets up stream options, including usage inclusion if specified, and prepares extra keyword arguments for the model call. This consolidates parameters like extra query, metadata, and additional body content into a single dictionary for the API request. Dependencies include copy.

```python
        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        extra_kwargs = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = copy(model_settings.extra_query)
        if model_settings.metadata:
            extra_kwargs["metadata"] = copy(model_settings.metadata)
        if model_settings.extra_body and isinstance(model_settings.extra_body, dict):
            extra_kwargs.update(model_settings.extra_body)
```

--------------------------------

### Get Workflow Name (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/traces

Retrieves the human-readable name of a workflow trace. This name should be descriptive and is used for grouping and filtering in dashboards. It is returned as a string.

```python
name: str

```

--------------------------------

### Agent Start Event for Realtime Sessions (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/events

Defines the RealtimeAgentStartEvent dataclass, which is emitted when a new agent begins its execution within a realtime session. It includes the agent instance and common event information. This event is crucial for tracking agent lifecycles.

```python
@dataclass
class RealtimeAgentStartEvent:
    """A new agent has started."""

    agent: RealtimeAgent
    """The new agent."""

    info: RealtimeEventInfo
    """Common info for all events, such as the context."""

    type: Literal["agent_start"] = "agent_start"

```

--------------------------------

### Get Model

Source: https://openai.github.io/openai-agents-python/ja/ref/models/interface

This endpoint retrieves a specific language model by its name from the model provider. It is a fundamental operation for selecting and utilizing different models within the system.

```APIDOC
## GET /get_model

### Description
Get a model by name from the model provider.

### Method
GET

### Endpoint
/get_model

### Parameters
#### Query Parameters
- **model_name** (str | None) - Required - The name of the model to get.

### Request Example
```json
{
  "model_name": "gpt-4"
}
```

### Response
#### Success Response (200)
- **model** (Model) - The model object.

#### Response Example
```json
{
  "model": { /* ... model details ... */ }
}
```
```

--------------------------------

### Get Function Tools from Single MCP Server (Python)

Source: https://openai.github.io/openai-agents-python/ref/mcp/util

This Python class method fetches all function tools from a specific MCP server. It utilizes a context manager (`mcp_tools_span`) for tracing and then converts the retrieved MCP tools into the Agents SDK `Tool` format. Dependencies include `MCPServer`, `RunContextWrapper`, `AgentBase`, `Tool`, and the `to_function_tool` method.

```python
import functools
import json
import logging
from typing import Any

from openai.types.beta.tools import FunctionTool
from pydantic import BaseModel

# Assuming MCPServer, RunContextWrapper, AgentBase, Tool, mcp_tools_span, ensure_strict_json_schema are defined elsewhere

class MCPTool(BaseModel):
    name: str
    description: str | None
    inputSchema: dict

class MCPServer:
    name: str
    def __init__(self, name: str):
        self.name = name
    async def list_tools(self, run_context: Any, agent: Any) -> list[MCPTool]:
        # Dummy implementation for example
        return []

class Tool:
    name: str
    description: str
    params_json_schema: dict
    on_invoke_tool: callable
    strict_json_schema: bool

class FunctionSpanData:
    output: str | None = None
    mcp_data: dict | None = None

class SpanData:
    pass

class CurrentSpan:
    span_data: SpanData | None = None

def get_current_span() -> CurrentSpan | None:
    # Dummy implementation for example
    return None

logger = logging.getLogger(__name__)

class UserError(Exception):
    pass

class AgentBase:
    pass

class RunContextWrapper[T]:
    pass

class MCPUtil:
    @classmethod
    async def get_function_tools(
        cls, server: MCPServer, convert_schemas_to_strict: bool, run_context: RunContextWrapper[Any], agent: AgentBase
    ) -> list[Tool]:
        """Get all function tools from a single MCP server."""

        with mcp_tools_span(server=server.name) as span:
            tools = await server.list_tools(run_context, agent)
            span.span_data.result = [tool.name for tool in tools]

        return [cls.to_function_tool(tool, server, convert_schemas_to_strict) for tool in tools]

    @classmethod
    def to_function_tool(
        cls, tool: MCPTool, server: MCPServer, convert_schemas_to_strict: bool
    ) -> FunctionTool:
        invoke_func = functools.partial(cls.invoke_mcp_tool, server, tool)
        schema, is_strict = tool.inputSchema, False

        if "properties" not in schema:
            schema["properties"] = {}

        if convert_schemas_to_strict:
            try:
                schema = ensure_strict_json_schema(schema)
                is_strict = True
            except Exception as e:
                logger.info(f"Error converting MCP schema to strict mode: {e}")

        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            params_json_schema=schema,
            on_invoke_tool=invoke_func,
            strict_json_schema=is_strict,
        )

    @classmethod
    def invoke_mcp_tool(cls, server: MCPServer, tool: MCPTool, **kwargs) -> str:
        # Dummy implementation for example
        print(f"Invoking tool {tool.name} on server {server.name} with args: {kwargs}")
        return json.dumps({})

    @classmethod
    async def get_all_function_tools(
        cls, servers: list[MCPServer], convert_schemas_to_strict: bool, run_context: RunContextWrapper[Any], agent: AgentBase
    ) -> list[Tool]:
        """Get all function tools from a list of MCP servers."""
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(
                server, convert_schemas_to_strict, run_context, agent
            )
            server_tool_names = {tool.name for tool in server_tools}
            if len(server_tool_names & tool_names) > 0:
                raise UserError(
                    f"Duplicate tool names found across MCP servers: "
                    f"{server_tool_names & tool_names}"
                )
            tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools

# Dummy definitions for functions used in the snippet
def mcp_tools_span(server: str):
    class MockSpanData:
        def __init__(self):
            self.result = []
    class MockSpan:
        def __init__(self):
            self.span_data = MockSpanData()
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return MockSpan()

def ensure_strict_json_schema(schema: dict) -> dict:
    return schema

```

--------------------------------

### Update Model Settings from Agent Configuration in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/session

Asynchronously retrieves and merges model settings from a combination of base settings, agent configurations (prompt, instructions, tools, handoffs), and starting settings. It handles optional agent parameters and disables tracing if configured.

```python
async def _get_updated_model_settings_from_agent(
 self,
 starting_settings: RealtimeSessionModelSettings | None,
 agent: RealtimeAgent,
) -> RealtimeSessionModelSettings:
 # Start with the merged base settings from run and model configuration.
 updated_settings = self._base_model_settings.copy()

 if agent.prompt is not None:
 updated_settings["prompt"] = agent.prompt

 instructions, tools, handoffs = await asyncio.gather(
 agent.get_system_prompt(self._context_wrapper),
 agent.get_all_tools(self._context_wrapper),
 self._get_handoffs(agent, self._context_wrapper),
 )
 updated_settings["instructions"] = instructions or ""
 updated_settings["tools"] = tools or []
 updated_settings["handoffs"] = handoffs or []

 # Apply starting settings (from model config) next
 if starting_settings:
 updated_settings.update(starting_settings)

 disable_tracing = self._run_config.get("tracing_disabled", False)
 if disable_tracing:
 updated_settings["tracing"] = None

 return updated_settings
```

--------------------------------

### Get State Options with Consistency and Concurrency - Python

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/dapr_session

This function retrieves StateOptions configured with strong or eventual consistency and optional concurrency settings. It uses Dapr client to set consistency levels based on internal configuration.

```Python
def _get_state_options(self, *, concurrency: Concurrency | None = None) -> StateOptions | None:
        """Get StateOptions configured with consistency and optional concurrency."""
        options_kwargs: dict[str, Any] = {}
        if self._consistency == DAPR_CONSISTENCY_STRONG:
            options_kwargs["consistency"] = Consistency.strong
        elif self._consistency == DAPR_CONSISTENCY_EVENTUAL:
            options_kwargs["consistency"] = Consistency.eventual
        if concurrency is not None:
            options_kwargs["concurrency"] = concurrency
        if options_kwargs:
            return StateOptions(**options_kwargs)
        return None
```

--------------------------------

### Configure Tracing with Non-OpenAI Models (Python)

Source: https://openai.github.io/openai-agents-python/tracing

This code snippet demonstrates how to enable tracing for non-OpenAI models using the LiteLLM integration. It sets the tracing API key from an environment variable and initializes a LiteLLMModel with a specified model name and API key, which can then be used with an Agent.

```python
import os
from agents import set_tracing_export_api_key, Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

tracing_api_key = os.environ["OPENAI_API_KEY"]
set_tracing_export_api_key(tracing_api_key)

model = LitellmModel(
    model="your-model-name",
    api_key="your-api-key",
)

agent = Agent(
    name="Assistant",
    model=model,
)

```

--------------------------------

### Create Higher-Level Traces with `trace()` Context Manager (Python)

Source: https://openai.github.io/openai-agents-python/tracing

This snippet demonstrates how to group multiple agent runs into a single trace using the `trace()` context manager. This is useful for monitoring and debugging complex workflows where several agent interactions form a logical unit. The `trace()` function automatically starts and ends the trace when entering and exiting the `with` block.

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"):
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")

```

--------------------------------

### Get Prompt Abstract Method (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/mcp/server

Defines an abstract asynchronous method to fetch a specific prompt from the server by its name, optionally with arguments. It returns a `GetPromptResult`. This is used to retrieve and potentially pre-configure prompts for use.

```python
@abc.abstractmethod
async def get_prompt(
    self,
    name: str,
    arguments: dict[str, Any] | None = None
) -> GetPromptResult:
    """Get a specific prompt from the server."""
    pass

```

--------------------------------

### MultiProvider Model Name Parsing Logic (Python)

Source: https://openai.github.io/openai-agents-python/ko/ref/models/multi_provider

Implements helper methods to parse model names. `_get_prefix_and_model_name` splits a model name string into its prefix and the actual model name, handling cases with or without a '/' separator. `_create_fallback_provider` dynamically instantiates provider classes like LitellmProvider based on a given prefix.

```python
    def _get_prefix_and_model_name(self, model_name: str | None) -> tuple[str | None, str | None]:
        if model_name is None:
            return None, None
        elif "/" in model_name:
            prefix, model_name = model_name.split("/", 1)
            return prefix, model_name
        else:
            return None, model_name

    def _create_fallback_provider(self, prefix: str) -> ModelProvider:
        if prefix == "litellm":
            from ..extensions.models.litellm_provider import LitellmProvider

            return LitellmProvider()
        else:
            raise UserError(f"Unknown prefix: {prefix}")

```

--------------------------------

### List Tools from MCP Server API (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Asynchronously lists the tools available on the server. It may utilize a cache if enabled and not invalidated. This function requires the server session to be initialized. Returns a list of available tools.

```python
async def list_tools(
    self,
    run_context: RunContextWrapper[Any] | None = None,
    agent: AgentBase | None = None,
) -> list[MCPTool]:
    """List the tools available on the server."""
    if not self.session:
        raise UserError("Server not initialized. Make sure you call `connect()` first.")
    session = self.session
    assert session is not None

    # Return from cache if caching is enabled, we have tools, and the cache is not dirty
    if self.cache_tools_list and not self._cache_dirty and self._tools_list:
        tools = self._tools_list
    else:
        # Fetch the tools from the server
        result = await self._run_with_retries(lambda: session.list_tools())
        self._tools_list = result.tools
        self._cache_dirty = False
        tools = self._tools_list

    # Filter tools based on tool_filter
    filtered_tools = tools
    if self.tool_filter is not None:
        filtered_tools = await self._apply_tool_filter(filtered_tools, run_context, agent)
    return filtered_tools

```

--------------------------------

### Configure Realtime Input Audio Transcription

Source: https://openai.github.io/openai-agents-python/ja/ref/realtime/config

Sets up configuration for transcribing audio in realtime sessions. Allows specifying language, transcription model (e.g., 'gpt-4o-transcribe', 'whisper-1'), and an optional prompt to guide the transcription process. This configuration is essential for processing spoken input accurately.

```python
class RealtimeInputAudioTranscriptionConfig(TypedDict):
    """Configuration for audio transcription in realtime sessions."""

    language: NotRequired[str]
    """The language code for transcription."""

    model: NotRequired[Literal["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"] | str]
    """The transcription model to use."""

    prompt: NotRequired[str]
    """An optional prompt to guide transcription."""

```

--------------------------------

### TracingProcessor Initialization

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/processors

Initializes a tracing processor with various configuration options for queue size, batch size, schedule delay, and export trigger ratio.

```APIDOC
## TracingProcessor __init__

### Description
Initializes a tracing processor with configurable parameters for managing traced data.

### Method
__init__

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **exporter** (TracingExporter) - Required - The exporter to use.
- **max_queue_size** (int) - Optional - The maximum number of spans to store in the queue. Defaults to 8192.
- **max_batch_size** (int) - Optional - The maximum number of spans to export in a single batch. Defaults to 128.
- **schedule_delay** (float) - Optional - The delay between checks for new spans to export. Defaults to 5.0.
- **export_trigger_ratio** (float) - Optional - The ratio of the queue size at which we will trigger an export. Defaults to 0.7.

### Request Example
```python
{
  "exporter": "your_tracing_exporter_instance",
  "max_queue_size": 4096,
  "max_batch_size": 64,
  "schedule_delay": 10.0,
  "export_trigger_ratio": 0.8
}
```

### Response
#### Success Response (200)
This is a constructor, no direct response.

#### Response Example
None
```

--------------------------------

### MCP List Tools Span API

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

This API allows for the creation of a new MCP list tools span. The span needs to be managed manually (started and finished) or used within a context manager.

```APIDOC
## POST /api/tracing/mcp_tools_span

### Description
Create a new MCP list tools span. The span will not be started automatically, you should either do `with mcp_tools_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/api/tracing/mcp_tools_span

#### Query Parameters
- **server** (str | None) - Optional - The name of the MCP server.
- **result** (list[str] | None) - Optional - The result of the MCP list tools call.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded. Defaults to `False`.

### Request Example
```json
{
  "server": "mcp_server_1",
  "result": ["tool1", "tool2"],
  "span_id": "generated_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span_data** (MCPListToolsSpanData) - The data associated with the span.

#### Response Example
```json
{
  "span_data": {
    "server": "mcp_server_1",
    "result": ["tool1", "tool2"]
  }
}
```
```

--------------------------------

### Get Realtime Playback State

Source: https://openai.github.io/openai-agents-python/ko/ref/realtime/openai_realtime

Retrieves the current playback state, including the active item ID, content index, and elapsed milliseconds. It prioritizes the playback tracker and falls back to audio state tracking.

```python
def _get_playback_state(self) -> RealtimePlaybackState:
        if self._playback_tracker:
            return self._playback_tracker.get_state()

        if last_audio_item_id := self._audio_state_tracker.get_last_audio_item():
            item_id, item_content_index = last_audio_item_id
            audio_state = self._audio_state_tracker.get_state(item_id, item_content_index)
            if audio_state:
                elapsed_ms = (
                    datetime.now() - audio_state.initial_received_time
                ).total_seconds() * 1000
                return {
                    "current_item_id": item_id,
                    "current_item_content_index": item_content_index,
                    "elapsed_ms": elapsed_ms,
                }

        return {
            "current_item_id": None,
            "current_item_content_index": None,
            "elapsed_ms": None,
        }
```

--------------------------------

### Initialize MCP Server with SSE Transport (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Initializes a new MCP server using HTTP with SSE transport. Key parameters include server configuration (`params`), tool list caching (`cache_tools_list`), server naming (`name`), client session timeout (`client_session_timeout_seconds`), tool filtering (`tool_filter`), structured content usage (`use_structured_content`), retry attempts (`max_retry_attempts`), retry backoff (`retry_backoff_seconds_base`), and an optional message handler (`message_handler`).

```python
def __init__(
    self,
    params: MCPServerSseParams,
    cache_tools_list: bool = False,
    name: str | None = None,
    client_session_timeout_seconds: float | None = 5,
    tool_filter: ToolFilter = None,
    use_structured_content: bool = False,
    max_retry_attempts: int = 0,
    retry_backoff_seconds_base: float = 1.0,
    message_handler: MessageHandlerFnT | None = None,
):
    """Create a new MCP server based on the HTTP with SSE transport.

    Args:
        params: The params that configure the server. This includes the URL of the server,
            the headers to send to the server, the timeout for the HTTP request, and the
            timeout for the SSE connection.

        cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
            cached and only fetched from the server once. If `False`, the tools list will be
            fetched from the server on each call to `list_tools()`. The cache can be
            invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
            if you know the server will not change its tools list, because it can drastically
            improve latency (by avoiding a round-trip to the server every time).

        name: A readable name for the server. If not provided, we'll create one from the
            URL.

        client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
        tool_filter: The tool filter to use for filtering tools.
        use_structured_content: Whether to use `tool_result.structured_content` when calling an
            MCP tool. Defaults to False for backwards compatibility - most MCP servers still
            include the structured content in the `tool_result.content`, and using it by
            default will cause duplicate content. You can set this to True if you know the
            server will not duplicate the structured content in the `tool_result.content`.
        max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
            Defaults to no retries.
        retry_backoff_seconds_base: The base delay, in seconds, for exponential
            backoff between retries.
        message_handler: Optional handler invoked for session messages as delivered by the
            ClientSession.
    """
    pass
```

--------------------------------

### RealtimeAgent Class Definition and Initialization - Python

Source: https://openai.github.io/openai-agents-python/ref/realtime/agent

Defines the RealtimeAgent class, inheriting from AgentBase and supporting generic type TContext. It outlines supported and unsupported configurations for voice agents within a RealtimeSession. It also details parameters like instructions, prompt, handoffs, output_guardrails, and hooks.

```python
from typing import Any, Callable, Generic, TypeVar, cast, Awaitable
import dataclasses
from dataclasses import dataclass, field
import inspect

from .agent_base import AgentBase
from .types import MaybeAwaitable, RunContextWrapper


# Define placeholder types for clarity in the example
class Prompt: pass
class RealtimeAgentHooks: pass
class OutputGuardrail: pass
class Handoff: pass

logger = type('Logger', (object,), {'error': lambda self, msg: print(f'ERROR: {msg}')})() # Mock logger

TContext = TypeVar("TContext")


@dataclass
class RealtimeAgent(AgentBase, Generic[TContext]):
    """A specialized agent instance that is meant to be used within a `RealtimeSession` to build
    voice agents. Due to the nature of this agent, some configuration options are not supported
    that are supported by regular `Agent` instances. For example:
    - `model` choice is not supported, as all RealtimeAgents will be handled by the same model
      within a `RealtimeSession`.
    - `modelSettings` is not supported, as all RealtimeAgents will be handled by the same model
      within a `RealtimeSession`.
    - `outputType` is not supported, as RealtimeAgents do not support structured outputs.
    - `toolUseBehavior` is not supported, as all RealtimeAgents will be handled by the same model
      within a `RealtimeSession`.
    - `voice` can be configured on an `Agent` level; however, it cannot be changed after the first
      agent within a `RealtimeSession` has spoken.

    See `AgentBase` for base parameters that are shared with `Agent`s.
    """

    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None
    """The instructions for the agent. Will be used as the "system prompt" when this agent is
    invoked. Describes what the agent should do, and how it responds.

    Can either be a string, or a function that dynamically generates instructions for the agent. If
    you provide a function, it will be called with the context and the agent instance. It must
    return a string.
    """

    prompt: Prompt | None = None
    """A prompt object. Prompts allow you to dynamically configure the instructions, tools
    and other config for an agent outside of your code. Only usable with OpenAI models.
    """

    handoffs: list[RealtimeAgent[Any] | Handoff[TContext, RealtimeAgent[Any]]] = field(
        default_factory=list
    )
    """Handoffs are sub-agents that the agent can delegate to. You can provide a list of handoffs,
    and the agent can choose to delegate to them if relevant. Allows for separation of concerns and
    modularity.
    """

    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    """A list of checks that run on the final output of the agent, after generating a response.
    Runs only if the agent produces a final output.
    """

    hooks: RealtimeAgentHooks | None = None
    """A class that receives callbacks on various lifecycle events for this agent.
    """

    def clone(self, **kwargs: Any) -> "RealtimeAgent[TContext]":
        """Make a copy of the agent, with the given arguments changed. For example, you could do:
        ```
        new_agent = agent.clone(instructions="New instructions")
        ```
        """
        return dataclasses.replace(self, **kwargs)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            else:
                return cast(str, self.instructions(run_context, self))
        elif self.instructions is not None:
            logger.error(f"Instructions must be a string or a function, got {self.instructions}")

        return None

```

--------------------------------

### Agent Span Creation API

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

This API endpoint allows for the creation of a new agent span. The span needs to be explicitly started and finished by the user, or managed using a `with` statement. It is used for tracing agent activities and their interactions.

```APIDOC
## POST /agent_span

### Description
Creates a new agent span for tracing. The span is not automatically started; manual start/finish or a `with` statement is required.

### Method
POST

### Endpoint
/agent_span

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **name** (string) - Required - The name of the agent.
- **handoffs** (list[string] | null) - Optional - A list of agent names to which this agent could hand off control.
- **tools** (list[string] | null) - Optional - A list of tool names available to this agent.
- **output_type** (string | null) - Optional - The name of the output type produced by the agent.
- **span_id** (string | null) - Optional - The ID of the span. If not provided, one will be generated.
- **parent** (Trace | Span[Any] | null) - Optional - The parent span or trace. If not provided, the current trace/span will be used.
- **disabled** (boolean) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "name": "MyAgent",
  "handoffs": ["NextAgent"],
  "tools": ["ToolA", "ToolB"],
  "output_type": "Result",
  "span_id": "unique-span-id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[AgentSpanData]** (object) - The newly created agent span, containing span details and data.

#### Response Example
```json
{
  "span_id": "unique-span-id",
  "name": "MyAgent",
  "start_time": "2023-10-27T10:00:00Z",
  "end_time": null,
  "data": {
    "handoffs": ["NextAgent"],
    "tools": ["ToolA", "ToolB"],
    "output_type": "Result"
  }
}
```
```

--------------------------------

### Conditionally Enable Tools with `is_enabled` Parameter

Source: https://openai.github.io/openai-agents-python/tools

Illustrates how to conditionally enable or disable agent tools at runtime using the `is_enabled` parameter. This example shows enabling a French tool only when a specific language preference is detected, using a callable function that checks the context.

```python
import asyncio
from agents import Agent, AgentBase, Runner, RunContextWrapper
from pydantic import BaseModel

class LanguageContext(BaseModel):
    language_preference: str = "french_spanish"

def french_enabled(ctx: RunContextWrapper[LanguageContext], agent: AgentBase) -> bool:
    """Enable French for French+Spanish preference."""
    return ctx.context.language_preference == "french_spanish"

# Create specialized agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You respond in Spanish. Always reply to the user's question in Spanish.",
)

french_agent = Agent(
    name="french_agent",
    instructions="You respond in French. Always reply to the user's question in French.",
)

# Create orchestrator with conditional tools
orchestrator = Agent(
    name="orchestrator",
    instructions=(
        "You are a multilingual assistant. You use the tools given to you to respond to users. "
        "You must call ALL available tools to provide responses in different languages. "
        "You never respond in languages yourself, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="respond_spanish",
            tool_description="Respond to the user's question in Spanish",
            is_enabled=True,  # Always enabled
        ),
        french_agent.as_tool(
            tool_name="respond_french",
            tool_description="Respond to the user's question in French",
            is_enabled=french_enabled,
        ),
    ],
)

async def main():
    context = RunContextWrapper(LanguageContext(language_preference="french_spanish"))
    result = await Runner.run(orchestrator, "How are you?", context=context.context)
    print(result.final_output)

asyncio.run(main())
```

--------------------------------

### Update Model Settings from Agent Configuration (Python)

Source: https://openai.github.io/openai-agents-python/ref/realtime/session

Retrieves and merges settings from an agent, including prompt, instructions, tools, and handoffs, with base model settings. It handles asynchronous fetching of agent data and applies optional starting settings. Dependencies include 'asyncio', 'RealtimeAgent', and 'RunContextWrapper'.

```python
    async def _get_updated_model_settings_from_agent(
        self,
        starting_settings: RealtimeSessionModelSettings | None,
        agent: RealtimeAgent,
    ) -> RealtimeSessionModelSettings:
        # Start with the merged base settings from run and model configuration.
        updated_settings = self._base_model_settings.copy()

        if agent.prompt is not None:
            updated_settings["prompt"] = agent.prompt

        instructions, tools, handoffs = await asyncio.gather(
            agent.get_system_prompt(self._context_wrapper),
            agent.get_all_tools(self._context_wrapper),
            self._get_handoffs(agent, self._context_wrapper),
        )
        updated_settings["instructions"] = instructions or ""
        updated_settings["tools"] = tools or []
        updated_settings["handoffs"] = handoffs or []

        # Apply starting settings (from model config) next
        if starting_settings:
            updated_settings.update(starting_settings)

        disable_tracing = self._run_config.get("tracing_disabled", False)
        if disable_tracing:
            updated_settings["tracing"] = None

        return updated_settings
```

--------------------------------

### Get Response Format Configuration in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

Determines the response format configuration based on the provided output schema. Returns 'omit' if the schema is None or indicates plain text output. Otherwise, it returns a JSON schema configuration for structured output.

```python
class Converter:
    @classmethod
    def get_response_format(
        cls,
        output_schema: AgentOutputSchemaBase | None
    ) -> ResponseTextConfigParam | Omit:
        if output_schema is None or output_schema.is_plain_text():
            return omit
        else:
            return {
                "format": {
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": output_schema.json_schema(),
                    "strict": output_schema.is_strict_json_schema(),
                }
            }
```

--------------------------------

### gen_trace_id: Generate Trace ID (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

The gen_trace_id method is responsible for generating a unique trace identifier. This abstract method is crucial for starting new traces within the TraceProvider.

```python
@abstractmethod
def gen_trace_id(self) -> str:
    """Generate a new trace identifier."""

```

--------------------------------

### MCPServerSse: Initialize HTTP SSE MCP Server (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/mcp/server

Initializes the MCPServerSse, a server implementation for the Model Context Protocol (MCP) using HTTP with Server-Sent Events (SSE) transport. It takes parameters for server configuration, caching, session timeouts, tool filtering, structured content usage, and retry logic. Dependencies include MCPServerSseParams and various types for session handling. The primary input is the MCPServerSseParams object containing server details.

```python
class MCPServerSse(_MCPServerWithClientSession):
    """MCP server implementation that uses the HTTP with SSE transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse)
    for details.
    """

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new MCP server based on the HTTP with SSE transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, and the
                timeout for the SSE connection.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.

            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
        """
        super().(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
        )

        self.params = params
        self._name = name or f"sse: {self.params['url']}"

```

--------------------------------

### Get Specific Prompt from MCP Server API (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/server

Asynchronously retrieves a specific prompt from the server by its name, optionally with arguments. Requires an active server session. Returns a GetPromptResult object. Raises UserError if the server is not initialized.

```python
async def get_prompt(
    self,
    name: str,
    arguments: dict[str, Any] | None = None
) -> GetPromptResult:
    """Get a specific prompt from the server."""
    if not self.session:
        raise UserError("Server not initialized. Make sure you call `connect()` first.")

    return await self.session.get_prompt(name, arguments)

```

--------------------------------

### MCP Tools Span Creation

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Creates a new span for MCP list tools. This span requires manual start and finish calls or can be used with a `with` statement. It captures information about the MCP server and the results of the list tools call.

```APIDOC
## POST /v1/tracing/mcp_tools_span

### Description
Create a new MCP list tools span. The span will not be started automatically; you should either use `with mcp_tools_span() ...` or call `span.start()` and `span.finish()` manually.

### Method
POST

### Endpoint
`/v1/tracing/mcp_tools_span`

### Parameters
#### Request Body
- **server** (str | None) - Optional - The name of the MCP server.
- **result** (list[str] | None) - Optional - The result of the MCP list tools call.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated. It is recommended to use `util.gen_span_id()` for correctly formatted IDs.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used as the parent.
- **disabled** (bool) - Optional - If True, a Span will be returned but not recorded. Defaults to False.

### Request Example
```json
{
  "server": "mcp_server_1",
  "result": ["tool1", "tool2"],
  "span_id": "generated_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **span** (Span[MCPListToolsSpanData]) - The created MCP list tools span object.

#### Response Example
```json
{
  "span": "<span_object_representation>"
}
```
```

--------------------------------

### Handle Tool Calls in Python

Source: https://openai.github.io/openai-agents-python/ref/realtime/session

Manages the execution of tool calls initiated by the model. It resolves tools and handoffs, dispatches tool start and end events, and invokes the actual tool logic. Tool results are sent back to the model.

```python
async def _handle_tool_call(
        self,
        event: RealtimeModelToolCallEvent,
        *, 
        agent_snapshot: RealtimeAgent | None = None,
    ) -> None:
        """Handle a tool call event."""
        agent = agent_snapshot or self._current_agent
        tools, handoffs = await asyncio.gather(
            agent.get_all_tools(self._context_wrapper),
            self._get_handoffs(agent, self._context_wrapper),
        )
        function_map = {tool.name: tool for tool in tools if isinstance(tool, FunctionTool)}
        handoff_map = {handoff.tool_name: handoff for handoff in handoffs}

        if event.name in function_map:
            await self._put_event(
                RealtimeToolStart(
                    info=self._event_info,
                    tool=function_map[event.name],
                    agent=agent,
                    arguments=event.arguments,
                )
            )

            func_tool = function_map[event.name]
            tool_context = ToolContext(
                context=self._context_wrapper.context,
                usage=self._context_wrapper.usage,
                tool_name=event.name,
                tool_call_id=event.call_id,
                tool_arguments=event.arguments,
            )
            result = await func_tool.on_invoke_tool(tool_context, event.arguments)

            await self._model.send_event(
                RealtimeModelSendToolOutput(
                    tool_call=event, output=str(result), start_response=True
                )
            )

            await self._put_event(
                RealtimeToolEnd(
                    info=self._event_info,
                    tool=func_tool,
                    output=result,
                    agent=agent,
                    arguments=event.arguments,
                )
            )

```

--------------------------------

### Python Basic Trace Usage

Source: https://openai.github.io/openai-agents-python/zh/ref/tracing

Illustrates the basic usage of the `trace` context manager in Python for an order processing workflow. The `with` statement ensures that the trace is properly entered and exited, handling any exceptions that may occur during the process. This basic setup is useful for tracking the execution of distinct operations.

```python
from opentelemetry_resamplers.trace import trace

# Basic trace usage
with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)
```

--------------------------------

### Manage Realtime Conversation Turn Events in Python

Source: https://openai.github.io/openai-agents-python/ref/realtime/openai_realtime

Handles events related to the start and end of conversational turns. It tracks whether a response is ongoing and emits corresponding events to signal the beginning and conclusion of assistant utterances. This helps in managing the flow of interaction.

```python
elif parsed.type == "response.created":
    self._ongoing_response = True
    await self._emit_event(RealtimeModelTurnStartedEvent())
elif parsed.type == "response.done":
    self._ongoing_response = False
    await self._emit_event(RealtimeModelTurnEndedEvent())
```

--------------------------------

### Create Database Tables and Indexes (SQL)

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Defines the SQL statements for creating necessary tables (message_structure, turn_usage) and indexes (idx_structure_branch_seq, idx_turn_usage_session_turn) to efficiently manage and query conversation data based on session, branch, and sequence.

```sql
CREATE TABLE IF NOT EXISTS message_structure (
            session_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            user_turn_number INTEGER NOT NULL,
            sequence_number INTEGER NOT NULL
        )
        
```

```sql
CREATE INDEX IF NOT EXISTS idx_structure_branch_seq
            ON message_structure(session_id, branch_id, sequence_number)
        
```

```sql
CREATE INDEX IF NOT EXISTS idx_turn_usage_session_turn
            ON turn_usage(session_id, branch_id, user_turn_number)
        
```

--------------------------------

### Span Properties API

Source: https://openai.github.io/openai-agents-python/ref/tracing/spans

This section details the properties available for a trace span, including identifiers, associated data, parent span information, error status, and timestamps for start and end times.

```APIDOC
## Span Properties

### Description
Provides access to various attributes of a tracing span, such as its unique identifiers, associated data, parent span, error status, and timing information.

### Properties

#### trace_id
- **Type**: `str`
- **Description**: The ID of the trace this span belongs to. Unique identifier of the parent trace.

#### span_id
- **Type**: `str`
- **Description**: Unique identifier for this span within its trace.

#### span_data
- **Type**: `TSpanData`
- **Description**: Operation-specific data for this span (e.g., LLM generation data).

#### parent_id
- **Type**: `str | None`
- **Description**: ID of the parent span, if any. None if this is a root span.

#### error
- **Type**: `SpanError | None`
- **Description**: Any error that occurred during span execution. Error details if an error occurred, None otherwise.

#### started_at
- **Type**: `str | None`
- **Description**: When the span started execution. ISO format timestamp of span start, None if not started.

#### ended_at
- **Type**: `str | None`
- **Description**: When the span finished execution. ISO format timestamp of span end, None if not finished.
```

--------------------------------

### Create Communication Streams for MCPServerSse in Python

Source: https://openai.github.io/openai-agents-python/ref/mcp/server

Defines the `create_streams` method for the MCPServerSse class, responsible for establishing the communication streams using the SSE client. It returns a context manager yielding the receive and send streams, along with a callback for getting the session ID. The SSE client is configured with the server's URL, headers, and timeouts.

```python
    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        """Create the streams for the server."""
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers", None),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
        )
```

--------------------------------

### Get Metadata for State Operations - Python

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/dapr_session

This function retrieves metadata for state operations, including the TTL (Time To Live) in seconds if configured. It returns a dictionary suitable for state store operations.

```Python
def _get_metadata(self) -> dict[str, str]:
        """Get metadata for state operations including TTL if configured."""
        metadata = {}
        if self._ttl is not None:
            metadata["ttlInSeconds"] = str(self._ttl)
        return metadata
```

--------------------------------

### Create Static Tool Filter

Source: https://openai.github.io/openai-agents-python/ja/ref/mcp/util

Creates a static tool filter configuration based on optional allowlist and blocklist of tool names.

```APIDOC
## POST /create_static_tool_filter

### Description
Creates a static tool filter from allowlist and blocklist parameters. This is a convenience function for creating a `ToolFilterStatic` object.

### Method
POST

### Endpoint
`/create_static_tool_filter`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **allowed_tool_names** (Optional[list[str]]) - Optional list of tool names to allow (whitelist). Defaults to None.
- **blocked_tool_names** (Optional[list[str]]) - Optional list of tool names to exclude (blacklist). Defaults to None.

### Request Example
```json
{
  "allowed_tool_names": ["tool1", "tool2"],
  "blocked_tool_names": ["tool3"]
}
```

### Response
#### Success Response (200)
- **tool_filter** (Optional[ToolFilterStatic]) - A `ToolFilterStatic` if any filtering is specified, otherwise None.

#### Response Example
```json
{
  "tool_filter": {
    "allowed_tool_names": ["tool1", "tool2"]
  }
}
```
```

--------------------------------

### Get Text-to-Speech Model (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_provider

Retrieves a text-to-speech (TTS) model by its name. If no model name is specified, it uses a default TTS model. This function relies on an active OpenAI client connection.

```python
def get_tts_model(self, model_name: str | None) -> TTSModel:
    """Get a text-to-speech model by name.

    Args:
        model_name: The name of the model to get.

    Returns:
        The text-to-speech model.
    """
    return OpenAITTSModel(model_name or DEFAULT_TTS_MODEL, self._get_client())

```

--------------------------------

### Get Model by Name - Python

Source: https://openai.github.io/openai-agents-python/ja/ref/models/interface

This abstract method defines the interface for retrieving a specific model by its name. It is part of the ModelProvider base class, which is responsible for managing and providing access to different models.

```python
import abc

# Assuming Model class is defined elsewhere
class Model:
    pass

class ModelProvider(abc.ABC):
    """The base interface for a model provider.

    Model provider is responsible for looking up Models by name.
    """

    @abc.abstractmethod
    def get_model(self, model_name: str | None) -> Model:
        """Get a model by name.

        Args:
            model_name: The name of the model to get.

        Returns:
            The model.
        """
        pass

```

--------------------------------

### Agent Span Creation

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing

The agent_span function is used to create a new agent span for tracing. Spans are not started automatically and require manual start/finish calls or usage within a 'with' statement.

```APIDOC
## POST /websites/openai_github_io_openai-agents-python/agent_span

### Description
Creates a new agent span for tracing purposes. The span needs to be manually started and finished, or used within a context manager.

### Method
POST

### Endpoint
/websites/openai_github_io_openai-agents-python/agent_span

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **name** (str) - Required - The name of the agent.
- **handoffs** (list[str] | None) - Optional - A list of agent names to which this agent could hand off control.
- **tools** (list[str] | None) - Optional - A list of tool names available to this agent.
- **output_type** (str | None) - Optional - The name of the output type produced by the agent.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, an ID will be generated.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, the current trace/span will be used.
- **disabled** (bool) - Optional - If True, the span will not be recorded. Defaults to False.

### Request Example
```json
{
  "name": "MyAgent",
  "handoffs": ["AnotherAgent"],
  "tools": ["Tool1", "Tool2"],
  "output_type": "string",
  "span_id": "generated-span-id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[AgentSpanData]** (object) - The newly created agent span.

#### Response Example
```json
{
  "span_id": "created-span-id",
  "name": "MyAgent",
  "handoffs": ["AnotherAgent"],
  "tools": ["Tool1", "Tool2"],
  "output_type": "string"
}
```
```

--------------------------------

### Initialize MCPServerStdio with Stdio Transport Parameters

Source: https://openai.github.io/openai-agents-python/ko/ref/mcp/server

Initializes an MCP server using stdio transport. It accepts configuration parameters including the command to run, arguments, environment variables, working directory, and text encoding. Dependencies include `_MCPServerWithClientSession` and `StdioServerParameters`. The constructor sets up server parameters and a readable name based on the command.

```python
class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server implementation that uses the stdio transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for
    details.
    """

    def __init__(
        self,
        params: MCPServerStdioParams,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new MCP server based on the stdio transport.

        Args:
            params: The params that configure the server. This includes the command to run to
                start the server, the args to pass to the command, the environment variables to
                set for the server, the working directory to use when spawning the process, and
                the text encoding used when sending/receiving messages to the server.
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).
            name: A readable name for the server. If not provided, we'll create one from the
                command.
            client_session_timeout_seconds: the read timeout passed to the MCP ClientSession.
            tool_filter: The tool filter to use for filtering tools.
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False for backwards compatibility - most MCP servers still
                include the structured content in the `tool_result.content`, and using it by
                default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to no retries.
            retry_backoff_seconds_base: The base delay, in seconds, for exponential
                backoff between retries.
            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.
        """
        super().__init__(
            cache_tools_list,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler=message_handler,
        )

        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get("encoding_error_handler", "strict"),
        )

        self._name = name or f"stdio: {self.params.command}"
```

--------------------------------

### Get Current Trace in Python

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/provider

Retrieves the currently active trace object from the execution context. If no trace is active, it returns None. This is essential for understanding the current state of tracing.

```python
def get_current_trace(self) -> Trace | None:
    """
    Returns the currently active trace, if any.
    """
    return Scope.get_current_trace()
```

--------------------------------

### Get Response from OpenAI API

Source: https://openai.github.io/openai-agents-python/ja/ref/models/openai_responses

Fetches a single response from the OpenAI API, handling system instructions, user input, model settings, tools, output schemas, handoffs, and tracing. It includes logging of the response and error handling. This method does not support streaming.

```python
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        with response_span(disabled=tracing.is_disabled()) as span_response:
            try:
                response = await self._fetch_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    stream=False,
                    prompt=prompt,
                )

                if _debug.DONT_LOG_MODEL_DATA:
                    logger.debug("LLM responded")
                else:
                    logger.debug(
                        "LLM resp:\n"
                        f"""{ 
                            json.dumps(
                                [x.model_dump() for x in response.output],
                                indent=2,
                                ensure_ascii=False,
                            )
                        }"""
                    )

                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.total_tokens,
                        input_tokens_details=response.usage.input_tokens_details,
                        output_tokens_details=response.usage.output_tokens_details,
                    )
                    if response.usage
                    else Usage()
                )

                if tracing.include_data():
                    span_response.span_data.response = response
                    span_response.span_data.input = input
            except Exception as e:
                span_response.set_error(
                    SpanError(
                        message="Error getting response",
                        data={
                            "error": str(e) if tracing.include_data() else e.__class__.__name__,
                        },
                    )
                )
                request_id = e.request_id if isinstance(e, APIStatusError) else None
                logger.error(f"Error getting response: {e}. (request_id: {request_id})")
                raise

        return ModelResponse(
            output=response.output,
            usage=usage,
            response_id=response.id,
        )
```

--------------------------------

### Initialize OpenAI Voice Model Provider (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/voice/models/openai_model_provider

Initializes the OpenAI voice model provider. It can be configured with an API key, base URL, organization, and project, or by providing an existing OpenAI client. This constructor is essential for setting up the connection to OpenAI's services.

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    organization: str | None = None,
    project: str | None = None,
) -> None:
    """Create a new OpenAI voice model provider.

    Args:
        api_key: The API key to use for the OpenAI client. If not provided, we will use the
            default API key.
        base_url: The base URL to use for the OpenAI client. If not provided, we will use the
            default base URL.
        openai_client: An optional OpenAI client to use. If not provided, we will create a new
            OpenAI client using the api_key and base_url.
        organization: The organization to use for the OpenAI client.
        project: The project to use for the OpenAI client.
    """
    if openai_client is not None:
        assert api_key is None and base_url is None,
            (
                "Don't provide api_key or base_url if you provide openai_client"
            )
        self._client: AsyncOpenAI | None = openai_client
    else:
        self._client = None
        self._stored_api_key = api_key
        self._stored_base_url = base_url
        self._stored_organization = organization
        self._stored_project = project

```

--------------------------------

### Get Current Time in ISO 8601 Format (Python)

Source: https://openai.github.io/openai-agents-python/ja/ref/tracing/util

Retrieves the current system time and formats it according to the ISO 8601 standard. This function relies on the trace provider to supply the time formatting functionality.

```python
def time_iso() -> str:
    """Return the current time in ISO 8601 format."""
    return get_trace_provider().time_iso()
```

--------------------------------

### Span Start Callback - Python

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing

Handles the event when a new span begins execution. This method is called synchronously and should execute quickly without blocking. It receives the `Span` object, which contains operation details and context. Spans are automatically nested within the current trace or span.

```python
@abc.abstractmethod
def on_span_start(self, span: "Span[Any]") -> None:
    """Called when a new span begins execution.

    Args:
        span: The span that started. Contains operation details and context.

    Notes:
        - Called synchronously on span start
        - Should return quickly to avoid blocking execution
        - Spans are automatically nested under current trace/span
    """
    pass

```

--------------------------------

### AdvancedSQLiteSession Constructor

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Initializes the AdvancedSQLiteSession with a session ID, database path, and options for table creation and logging.

```APIDOC
## AdvancedSQLiteSession Constructor

### Description
Initializes the AdvancedSQLiteSession with a session ID, database path, and options for table creation and logging.

### Method
`__init__`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **session_id** (str) - Required - The ID of the session
- **db_path** (str | Path) - Optional - The path to the SQLite database file. Defaults to `:memory:` for in-memory storage
- **create_tables** (bool) - Optional - Whether to create the structure tables. Defaults to `False`
- **logger** (Logger | None) - Optional - The logger to use. Defaults to the module logger
- **kwargs** - Optional - Additional keyword arguments to pass to the superclass

### Request Example
```python
AdvancedSQLiteSession(
    session_id="my_session",
    db_path=":memory:",
    create_tables=True,
    logger=my_logger
)
```

### Response
#### Success Response (Initialization)
This method does not return a value but initializes the object.

#### Response Example
N/A
```

--------------------------------

### Implement Output Guardrail with Agent

Source: https://openai.github.io/openai-agents-python/guardrails

This code defines an output guardrail that uses a separate agent to check if the agent's response contains any mathematical content. It intercepts the output, runs it through the guardrail agent, and returns a GuardrailFunctionOutput to indicate if a tripwire was triggered. The guardrail is then applied to a customer support agent.

```python
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)
class MessageOutput(BaseModel): 
    response: str

class MathOutput(BaseModel): 
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent( 
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")

```

--------------------------------

### Get Function Tools from a Single MCP Server (Python)

Source: https://openai.github.io/openai-agents-python/ref/mcp/util

Retrieves all function tools from a single MCP server. It uses a tracing span for monitoring the tool retrieval process and then converts each retrieved MCP tool into an Agents SDK FunctionTool format. This is a foundational method for fetching tools from a specific MCP endpoint.

```python
class MCPUtil:
    """Set of utilities for interop between MCP and Agents SDK tools."""

    @classmethod
    async def get_function_tools(
        cls,
        server: "MCPServer",
        convert_schemas_to_strict: bool,
        run_context: RunContextWrapper[Any],
        agent: "AgentBase",
    ) -> list[Tool]:
        """Get all function tools from a single MCP server."""

        with mcp_tools_span(server=server.name) as span:
            tools = await server.list_tools(run_context, agent)
            span.span_data.result = [tool.name for tool in tools]

        return [cls.to_function_tool(tool, server, convert_schemas_to_strict) for tool in tools]

```

--------------------------------

### Get Turn Usage (Python)

Source: https://openai.github.io/openai-agents-python/ref/extensions/memory/advanced_sqlite_session

Asynchronously retrieves the total token usage for a given turn from the session. It calculates input and output token details by processing rows from a database query. This is useful for monitoring and analyzing API costs.

```python
async def get_turn_usage(self) -> Union[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve the total token usage for this turn."""

    def _get_turn_usage_sync():
        results = []
        conn = self._get_connection()
        with self._lock if self._is_memory_db else threading.Lock():
            cursor = conn.execute(
                f"""
                SELECT messages.created_at, messages.created_by, messages.message, messages.session_id, SUM(tokens.n), SUM(tokens.prompt_tokens), SUM(tokens.completion_tokens)
                FROM messages
                INNER JOIN tokens ON messages.session_id = tokens.session_id AND messages.created_at = tokens.created_at
                WHERE messages.session_id = ? AND messages.created_at BETWEEN ? AND ?
                GROUP BY messages.created_at, messages.created_by, messages.message, messages.session_id
                ORDER BY messages.created_at DESC
                """,
                (self.session_id, self.start_turn_timestamp, self.end_turn_timestamp),
            )

            for row in cursor.fetchall():
                input_details = {
                    "total_tokens": row[5],
                }
                output_details = {
                    "total_tokens": row[6],
                }
                results.append(
                    {
                        "created_at": row[0],
                        "created_by": row[1],
                        "message": row[2],
                        "session_id": row[3],
                        "total_tokens": row[4],
                        "input_tokens_details": input_details,
                        "output_tokens_details": output_details,
                    }
                )
        return results

    result = await asyncio.to_thread(_get_turn_usage_sync)

    return cast(Union[list[dict[str, Any]], dict[str, Any]], result)
```

--------------------------------

### Get Current Mapping from MultiProviderMap

Source: https://openai.github.io/openai-agents-python/ko/ref/models/multi_provider

The `get_mapping` method returns a shallow copy of the current dictionary that maps model name prefixes to `ModelProvider` objects. This prevents external modification of the internal mapping.

```python
def get_mapping(self) -> dict[str, ModelProvider]:
    """Returns a copy of the current prefix -> ModelProvider mapping."""
    return self._mapping.copy()
```

--------------------------------

### Configure Agent Model with Extra OpenAI API Arguments

Source: https://openai.github.io/openai-agents-python/models

Illustrates how to pass additional, provider-specific arguments to the OpenAI API when configuring an agent. This is achieved using the `extra_args` parameter within `ModelSettings`, enabling the use of parameters like `service_tier` and `user`.

```python
from agents import Agent, ModelSettings

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4.1",
    model_settings=ModelSettings(
        temperature=0.1,
        extra_args={"service_tier": "flex", "user": "user_12345"},
    ),
)


```

--------------------------------

### Check if GPT-5 Reasoning Settings Required (Python)

Source: https://openai.github.io/openai-agents-python/ref/models/default_models

Determines if a given model name corresponds to a GPT-5 model that requires specific reasoning settings. It checks if the model name starts with 'gpt-5' but excludes 'gpt-5-chat-latest'.

```Python
def gpt_5_reasoning_settings_required(model_name: str) -> bool:
    """
    Returns True if the model name is a GPT-5 model and reasoning settings are required.
    """
    if model_name.startswith("gpt-5-chat"):
        # gpt-5-chat-latest does not require reasoning settings
        return False
    # matches any of gpt-5 models
    return model_name.startswith("gpt-5")
```

--------------------------------

### Custom Span Creation

Source: https://openai.github.io/openai-agents-python/ko/ref/tracing/create

The `custom_span` function allows you to create and manage custom spans for detailed tracing of operations within your application. Spans are not started automatically and require manual management or use within a `with` statement.

```APIDOC
## POST /custom_span

### Description
Create a new custom span, to which you can add your own metadata. The span will not be started automatically, you should either do `with custom_span() ...` or call `span.start()` + `span.finish()` manually.

### Method
POST

### Endpoint
/custom_span

### Parameters
#### Query Parameters
- **name** (str) - Required - The name of the custom span.
- **data** (dict[str, Any] | None) - Optional - Arbitrary structured data to associate with the span.
- **span_id** (str | None) - Optional - The ID of the span. If not provided, we will generate an ID. We recommend using `util.gen_span_id()` to generate a span ID, to guarantee that IDs are correctly formatted.
- **parent** (Trace | Span[Any] | None) - Optional - The parent span or trace. If not provided, we will automatically use the current trace/span as the parent.
- **disabled** (bool) - Optional - If True, we will return a Span but the Span will not be recorded.

### Request Example
```json
{
  "name": "my_custom_operation",
  "data": {
    "key1": "value1",
    "key2": 123
  },
  "span_id": "optional_span_id",
  "parent": null,
  "disabled": false
}
```

### Response
#### Success Response (200)
- **Span[CustomSpanData]** (object) - The newly created custom span.
```

--------------------------------

### Merge System Content (Python)

Source: https://openai.github.io/openai-agents-python/zh/ref/realtime/session

Merges incoming system content into existing history, prioritizing existing non-empty text when the incoming content is empty. This ensures that essential system instructions or configurations are not lost due to empty updates.

```python
                    elif event.role == "system" and existing_item.role == "system":
                        system_existing_content = existing_item.content
                        system_incoming = event.content
                        # Prefer existing non-empty text when incoming is empty
                        system_new_content: list[InputText] = []
                        for idx, sc in enumerate(system_incoming):
                            if idx >= len(system_existing_content):
```

--------------------------------

### Get Tool Usage OpenAI Agents Python

Source: https://openai.github.io/openai-agents-python/ja/ref/extensions/memory/advanced_sqlite_session

Fetches tool usage statistics for a given branch or the entire session. It returns a list of tuples, each containing the tool name, its usage count, and the corresponding turn number. This function is designed to work with asynchronous execution.

```python
async def get_tool_usage(self, branch_id: str | None = None) -> list[tuple[str, int, int]]:
        """Get all tool usage by turn for specified branch.

        Args:
            branch_id: Branch to get tool usage from (current branch if None).

        Returns:
            List of tuples containing (tool_name, usage_count, turn_number).
        """
        if branch_id is None:
            branch_id = self._current_branch_id

        def _get_tool_usage_sync():
            """Synchronous helper to get tool usage statistics."""
            conn = self._get_connection()
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    """
                    SELECT tool_name, COUNT(*), user_turn_number
                    FROM message_structure
                    WHERE session_id = ? AND branch_id = ? AND message_type IN (
                        'tool_call', 'function_call', 'computer_call', 'file_search_call',
                        'web_search_call', 'code_interpreter_call', 'custom_tool_call',
                        'mcp_call', 'mcp_approval_request'
                    )
                    GROUP BY tool_name, user_turn_number
                    ORDER BY user_turn_number
                """,
                    (self.session_id, branch_id),
                )
                return cursor.fetchall()

        return await asyncio.to_thread(_get_tool_usage_sync)

```