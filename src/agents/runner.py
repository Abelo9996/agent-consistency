"""
Unified agent runner that supports multiple LLM providers with tool calling.

Each run returns a structured trace of all tool calls and the final response.
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    """A single tool call made by the agent."""
    step: int
    tool_name: str
    arguments: dict
    result: str
    timestamp: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace of an agent run."""
    task_id: str
    model: str
    run_index: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str = ""
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def tool_sequence(self) -> list[str]:
        """Just the tool names in order."""
        return [tc.tool_name for tc in self.tool_calls]

    @property
    def tool_args_sequence(self) -> list[tuple[str, dict]]:
        """Tool names with their arguments."""
        return [(tc.tool_name, tc.arguments) for tc in self.tool_calls]


SYSTEM_PROMPT = """You are a helpful assistant with access to tools. Use them to complete tasks.

Rules:
- Use the provided tools to accomplish the task
- Call tools as needed, then provide a final response
- Be thorough but efficient — use the minimum tools necessary
- If a task is ambiguous, make reasonable assumptions and proceed"""


MAX_ITERATIONS = 10


def run_agent_openai(model: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent loop using OpenAI's API."""
    from openai import OpenAI
    client = OpenAI()

    trace = AgentTrace(task_id="", model=model, run_index=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    start = time.time()
    step = 0

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schemas if tools_schemas else None,
            temperature=1.0,  # Default temperature — intentional, we're measuring natural variance
        )
        choice = response.choices[0]
        trace.total_tokens += response.usage.total_tokens if response.usage else 0

        if choice.message.tool_calls:
            # Process each tool call
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                step += 1
                args = json.loads(tc.function.arguments)
                result = execute_fn(tc.function.name, args)
                trace.tool_calls.append(ToolCall(
                    step=step,
                    tool_name=tc.function.name,
                    arguments=args,
                    result=result,
                    timestamp=time.time() - start,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            trace.final_response = choice.message.content or ""
            break

    trace.latency_ms = (time.time() - start) * 1000
    return trace


def run_agent_anthropic(model: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent loop using Anthropic's API."""
    import anthropic
    client = anthropic.Anthropic()

    # Convert OpenAI tool format to Anthropic format
    anthropic_tools = []
    for t in tools_schemas:
        anthropic_tools.append({
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        })

    trace = AgentTrace(task_id="", model=model, run_index=0)
    messages = [{"role": "user", "content": task}]
    start = time.time()
    step = 0

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=anthropic_tools,
            temperature=1.0,
        )
        trace.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # Check if there are tool uses in the response
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tu in tool_uses:
                step += 1
                result = execute_fn(tu.name, tu.input)
                trace.tool_calls.append(ToolCall(
                    step=step,
                    tool_name=tu.name,
                    arguments=tu.input,
                    result=result,
                    timestamp=time.time() - start,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            trace.final_response = "\n".join(text_blocks)
            break

        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            trace.final_response = "\n".join(text_blocks)
            break

    trace.latency_ms = (time.time() - start) * 1000
    return trace


def run_agent_google(model: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent loop using Google's Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Convert to Gemini tool format
    function_declarations = []
    for t in tools_schemas:
        function_declarations.append(genai.protos.FunctionDeclaration(
            name=t["function"]["name"],
            description=t["function"]["description"],
            parameters=t["function"]["parameters"],
        ))

    gemini_tools = genai.protos.Tool(function_declarations=function_declarations)
    gemini_model = genai.GenerativeModel(model, tools=[gemini_tools], system_instruction=SYSTEM_PROMPT)

    trace = AgentTrace(task_id="", model=model, run_index=0)
    chat = gemini_model.start_chat()
    start = time.time()
    step = 0

    response = chat.send_message(task)

    for _ in range(MAX_ITERATIONS):
        # Check for function calls
        fn_calls = []
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fn_calls.append(part)

        if not fn_calls:
            trace.final_response = response.text if response.text else ""
            break

        # Execute function calls
        responses = []
        for part in fn_calls:
            fc = part.function_call
            step += 1
            args = dict(fc.args) if fc.args else {}
            result = execute_fn(fc.name, args)
            trace.tool_calls.append(ToolCall(
                step=step,
                tool_name=fc.name,
                arguments=args,
                result=result,
                timestamp=time.time() - start,
            ))
            responses.append(genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=fc.name,
                    response={"result": json.loads(result)},
                )
            ))

        response = chat.send_message(genai.protos.Content(parts=responses))

    trace.latency_ms = (time.time() - start) * 1000
    if response.usage_metadata:
        trace.total_tokens = response.usage_metadata.total_token_count or 0
    return trace


def run_agent_together(model: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent loop using Together AI (for open-source models)."""
    from together import Together
    client = Together()

    trace = AgentTrace(task_id="", model=model, run_index=0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    start = time.time()
    step = 0

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schemas,
            temperature=1.0,
        )
        choice = response.choices[0]
        if response.usage:
            trace.total_tokens += response.usage.total_tokens

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                step += 1
                args = json.loads(tc.function.arguments)
                result = execute_fn(tc.function.name, args)
                trace.tool_calls.append(ToolCall(
                    step=step,
                    tool_name=tc.function.name,
                    arguments=args,
                    result=result,
                    timestamp=time.time() - start,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            trace.final_response = choice.message.content or ""
            break

    trace.latency_ms = (time.time() - start) * 1000
    return trace


def run_agent_openai_reasoning(model: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent loop using OpenAI's reasoning models (o1, o3-mini).
    These models don't support temperature or system messages."""
    from openai import OpenAI
    client = OpenAI()

    trace = AgentTrace(task_id="", model=model, run_index=0)
    # Reasoning models: no system message, use developer message instead
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    start = time.time()
    step = 0

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schemas if tools_schemas else None,
            # No temperature param for reasoning models
        )
        choice = response.choices[0]
        trace.total_tokens += response.usage.total_tokens if response.usage else 0

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                step += 1
                args = json.loads(tc.function.arguments)
                result = execute_fn(tc.function.name, args)
                trace.tool_calls.append(ToolCall(
                    step=step,
                    tool_name=tc.function.name,
                    arguments=args,
                    result=result,
                    timestamp=time.time() - start,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            trace.final_response = choice.message.content or ""
            break

    trace.latency_ms = (time.time() - start) * 1000
    return trace


# Provider routing
PROVIDERS = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4-turbo": ("openai", "gpt-4-turbo"),
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "gpt-4.1-nano": ("openai", "gpt-4.1-nano"),
    "o1": ("openai-reasoning", "o1"),
    "o3-mini": ("openai-reasoning", "o3-mini"),
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5-20250514"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    "gemini-2.0-flash": ("google", "gemini-2.0-flash"),
    "llama-3.3-70b": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "llama-3.1-70b": ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "llama-3.1-8b": ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
}

RUNNER_MAP = {
    "openai": run_agent_openai,
    "openai-reasoning": run_agent_openai_reasoning,
    "anthropic": run_agent_anthropic,
    "google": run_agent_google,
    "together": run_agent_together,
}


def run_agent(model_key: str, task: str, tools_schemas: list, execute_fn) -> AgentTrace:
    """Run an agent with the appropriate provider."""
    if model_key not in PROVIDERS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(PROVIDERS.keys())}")
    provider, model_id = PROVIDERS[model_key]
    runner = RUNNER_MAP[provider]
    return runner(model_id, task, tools_schemas, execute_fn)
