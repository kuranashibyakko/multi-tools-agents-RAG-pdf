import json
import os
import re
import time
from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

from workers.coder import run_coder_worker
from workers.critic import run_critic_worker
from workers.researcher import run_researcher_worker
from workers.web_search import run_web_search_worker

load_dotenv()
groq_client = Groq(timeout=25.0)
MODEL_NAME = "openai/gpt-oss-20b"

SUPERVISOR_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_researcher",
            "description": (
                "Search the local fruit juice PDF in ChromaDB. "
                "Use once for questions about the study's experimental juice findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Specific research inquiry for the local paper.",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_web_search",
            "description": (
                "Search the live internet for nutrition or external facts. "
                "Use clear, factual query phrases focusing on one or two items with keywords like 'nutrition facts' or 'grams sugar' "
                "(e.g., 'Coca-Cola vs Pepsi sugar grams per 100ml nutrition facts')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Specific search query.",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_coder",
            "description": (
                "Write and execute Python scripts in the sandbox for math, conversions, or simulations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Python task to execute.",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_critic",
            "description": (
                "Audit a draft comparison, calculation, or findings against the original goal. "
                "Returns STATUS: PASSED or STATUS: NEEDS_REVISION."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The exact verbatim original user goal.",
                    },
                    "proposal": {
                        "type": "string",
                        "description": "The draft solution, comparison table, or findings to verify.",
                    },
                },
                "required": ["task", "proposal"],
            },
        },
    },
]

SUPERVISOR_TOOLS_MAP = {
    "delegate_to_researcher": run_researcher_worker,
    "delegate_to_web_search": run_web_search_worker,
    "delegate_to_coder": run_coder_worker,
    "delegate_to_critic": run_critic_worker,
}


def call_groq_with_retry(messages, tools=None, max_retries=4):
    """Executes a Groq completion with automatic backoff for 429 rate limits."""
    for attempt in range(max_retries):
        try:
            kwargs = {"model": MODEL_NAME, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return groq_client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            err_str = str(e)
            match = re.search(r"try again in (\d+\.?\d*)s", err_str)
            wait_time = float(match.group(1)) + 1.0 if match else 5.0
            print(f"\n[Groq 429 TPM Rate Limit]: Backing off for {wait_time:.1f}s...")
            time.sleep(wait_time)
        except BadRequestError as e:
            raise e
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2.0)
    raise RuntimeError("Exceeded maximum retries for Groq API call.")


def run_supervisor(user_request: str, max_iterations: int = 5, step_callback=None) -> str:
    """Orchestrates the multi-agent network with rate-limit protection and context pruning."""
    system_prompt = (
        "You are the Lead Supervisor of an autonomous multi-agent network.\n"
        "Workers:\n"
        "1. Researcher: Searches local fruit juice paper in ChromaDB.\n"
        "2. Web Search: Searches live internet for soda data.\n"
        "3. Coder: Runs Python calculations.\n"
        "4. Critic: Audits solutions.\n\n"
        "EFFICIENCY & TOKEN MANAGEMENT RULES:\n"
        "- Query the Researcher at most ONCE. If the paper lacks numeric values, accept that immediately.\n"
        "- Query Web Search at most ONCE by batching all soda queries into a single call (e.g. 'sugar content per 100ml Coca-Cola Sprite').\n"
        "- Combine the findings into the requested comparison table and call delegate_to_critic ONCE.\n"
        "- Pass the original user request as 'task' to delegate_to_critic.\n"
        "- When the Critic responds with PASSED, output the final synthesis immediately."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

    print(f"\n=======================================================")
    print(f"[Supervisor Received Goal]: {user_request}")
    print(f"=======================================================\n")

    for iteration in range(1, max_iterations + 1):
        print(f"--- [Supervisor Cycle {iteration}] ---")

        try:
            response = call_groq_with_retry(messages, tools=SUPERVISOR_TOOLS_SCHEMA)
        except Exception as e:
            err_msg = f"API Error: {str(e)}"
            if step_callback:
                step_callback("error", err_msg)
            return err_msg

        response_message = response.choices[0].message

        # Case A: Ready to synthesize
        if not response_message.tool_calls:
            print("[Supervisor]: All worker tasks resolved. Final synthesis ready.\n")
            if step_callback:
                step_callback("final", response_message.content)
            return response_message.content

        # Case B: Tool Dispatch
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            worker_function = tool_call.function.name
            raw_args = tool_call.function.arguments

            print(f"[Supervisor -> {worker_function}]: {raw_args}")

            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}

            if step_callback:
                step_callback(
                    "dispatch",
                    f"**Supervisor $\\rightarrow$ `{worker_function}`**\n```json\n{raw_args}\n```",
                )

            if worker_function in SUPERVISOR_TOOLS_MAP:
                worker_result = SUPERVISOR_TOOLS_MAP[worker_function](**parsed_args)
            else:
                worker_result = f"Error: Worker '{worker_function}' is not registered."

            print(f"[Worker Response]: {str(worker_result)[:140]}...\n")

            if step_callback:
                step_callback(
                    "observation",
                    f"**Observation from `{worker_function}`:**\n{worker_result}",
                )

            # Context pruning: truncate stored tool observation to 600 chars to avoid TPM bloat
            pruned_result = str(worker_result)
            if len(pruned_result) > 700:
                pruned_result = pruned_result[:700] + "\n...[truncated for context limits]"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": pruned_result,
            })

    # Fallback synthesis if iterations finish
    print("[Supervisor]: Synthesizing final answer from available context...")
    messages.append({
        "role": "user",
        "content": "Please synthesize the complete final response now based on all worker observations.",
    })
    try:
        final_call = call_groq_with_retry(messages)
        final_answer = final_call.choices[0].message.content or "Completed analysis."
    except Exception as e:
        final_answer = f"Completed analysis with available data: {str(e)}"

    if step_callback:
        step_callback("final", final_answer)
    return final_answer


if __name__ == "__main__":
    print("==================================================")
    print("Multi-Agent Supervisor Network Online")
    print("Active Personas: Supervisor, Researcher, Web Search, Coder, Critic")
    print("Type 'exit' or 'q' to shut down.")
    print("==================================================")

    while True:
        user_input = input("\nEnter Request: ").strip()
        if user_input.lower() in ("exit", "q"):
            break
        if not user_input:
            continue
        print(f"\nFINAL SYNTHESIS:\n{run_supervisor(user_input)}\n")