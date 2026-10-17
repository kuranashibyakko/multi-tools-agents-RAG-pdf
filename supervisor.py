from concurrent.futures import ThreadPoolExecutor
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
                "Search the local indexed PDF documents in ChromaDB. "
                "Use once for questions about local study experiments or uploaded papers."
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
                "Search the live internet for nutrition facts, benchmarks, or external knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Search query to look up.",
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
                "Write and execute Python scripts in the sandbox for math, conversions, "
                "simulations, or generating matplotlib charts/plots (saved as output_plot.png)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Python calculation, simulation, or matplotlib chart task to execute.",
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
                        "description": "The draft solution, summary statistics, or findings to verify.",
                    },
                },
                "required": ["task"],
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


def run_supervisor(
    user_request: str,
    max_iterations: int = 6,
    step_callback=None,
    chat_history: list | None = None,
) -> str:
    """Orchestrates the multi-agent network with sliding-window memory and parallel tool dispatch."""
    system_prompt = (
        "You are the Lead Supervisor of an autonomous multi-agent network.\n"
        "Workers:\n"
        "1. Researcher ('delegate_to_researcher'): Searches local PDFs in ChromaDB.\n"
        "2. Web Search ('delegate_to_web_search'): Searches internet for external facts/nutrition data.\n"
        "3. Coder ('delegate_to_coder'): Runs Python simulations and saves matplotlib plots to 'output_plot.png'.\n"
        "4. Critic ('delegate_to_critic'): Audits solutions.\n\n"
        "EXECUTION RULES:\n"
        "- You can emit multiple tool calls simultaneously (e.g., Researcher + Web Search) when tasks are independent.\n"
        "- Call delegate_to_coder ONCE for a simulation or plot. If it returns '[Coder Worker: Execution Succeeded]' "
        "and '[ARTIFACT CREATED]', do NOT re-run the Coder.\n"
        "- Next, call delegate_to_critic ONCE with both 'task' (original request) and 'proposal' (the summary stats and confirmation that the plot was created).\n"
        "- As soon as the Critic returns STATUS: PASSED, stop calling tools and output your final synthesis."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject sliding-window memory (last 4 messages, pruned to 400 chars each to protect TPM)
    if chat_history:
        for past_msg in chat_history[-4:]:
            role = past_msg.get("role", "user")
            content = str(past_msg.get("content", ""))[:400]
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_request})

    collected_observations = []

    for iteration in range(1, max_iterations + 1):
        try:
            response = call_groq_with_retry(messages, tools=SUPERVISOR_TOOLS_SCHEMA)
        except Exception as e:
            print(f"[Tool formatting fallback on cycle {iteration}: {e}]")
            break

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            if step_callback:
                step_callback("final", response_message.content)
            return response_message.content

        messages.append(response_message)

        # Prepare tool calls for execution
        parsed_calls = []
        for tool_call in response_message.tool_calls:
            worker_function = tool_call.function.name
            raw_args = tool_call.function.arguments
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"task": user_request}

            if worker_function == "delegate_to_critic" and not parsed_args.get("proposal"):
                parsed_args["proposal"] = "\n\n".join(collected_observations) or "Completed task."

            parsed_calls.append((tool_call.id, worker_function, parsed_args))

            if step_callback:
                step_callback(
                    "dispatch",
                    f"**Supervisor $\\rightarrow$ `{worker_function}`**\n```json\n{json.dumps(parsed_args, indent=2)}\n```",
                )

        # Execute tool calls concurrently using ThreadPoolExecutor
        def execute_single_tool(call_tuple):
            call_id, func_name, kwargs = call_tuple
            if func_name in SUPERVISOR_TOOLS_MAP:
                res = SUPERVISOR_TOOLS_MAP[func_name](**kwargs)
            else:
                res = f"Error: Worker '{func_name}' is not registered."
            return call_id, func_name, res

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(execute_single_tool, parsed_calls))

        for call_id, worker_function, worker_result in results:
            if step_callback:
                step_callback(
                    "observation",
                    f"**Observation from `{worker_function}`:**\n{worker_result}",
                )

            pruned_result = str(worker_result)
            if len(pruned_result) > 1500:
                pruned_result = pruned_result[:1500] + "\n...[truncated]"

            collected_observations.append(f"[{worker_function}]:\n{pruned_result}")

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": pruned_result,
            })

    # Clean, tool-free fallback synthesis
    compiled_notes = "\n\n---\n\n".join(collected_observations)
    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "You are the Lead Supervisor. Synthesize a complete, well-formatted final markdown report "
                "answering the user's request based on the worker execution logs below."
            ),
        },
        {
            "role": "user",
            "content": f"ORIGINAL REQUEST:\n{user_request}\n\nWORKER LOGS:\n{compiled_notes}",
        },
    ]

    try:
        final_call = call_groq_with_retry(synthesis_messages, tools=None)
        final_answer = final_call.choices[0].message.content or "Completed analysis."
    except Exception as e:
        final_answer = f"Completed analysis:\n\n{compiled_notes}\n\n(Note: {str(e)})"

    if step_callback:
        step_callback("final", final_answer)
    return final_answer