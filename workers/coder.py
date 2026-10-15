import os
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
from groq import Groq

# 1. Environment & Setup
load_dotenv()
groq_client = Groq()
MODEL_NAME = "openai/gpt-oss-20b"

# Dedicated directory for sandbox execution
SANDBOX_DIR = Path("./sandbox").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)


def execute_python_code(code: str, timeout_seconds: int = 10) -> dict:
    """Executes Python code in an isolated subprocess inside the sandbox folder."""
    script_path = SANDBOX_DIR / "temp_script.py"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(SANDBOX_DIR),
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Subprocess runner failed: {str(e)}",
            "return_code": -1,
        }


def run_coder_worker(task: str) -> str:
    """Coder persona: writes Python code for the task, runs it in the sandbox,
    and returns the execution output to the supervisor.
    """
    system_prompt = (
        "You are an expert Python programming worker in a multi-agent network.\n"
        "Your task is to write clean, complete, standalone Python code to accomplish the given goal.\n\n"
        "Rules:\n"
        "1. Output ONLY the raw executable Python code inside a single ```python ``` markdown block.\n"
        "2. Ensure the code prints its final results to stdout using print().\n"
        "3. Do not include interactive input() calls or external network requests.\n"
        "4. Keep the code safe, robust, and self-contained."
    )

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content or ""

    # Extract code between fences
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code = content.split("```")[1].split("```")[0].strip()
    else:
        code = content.strip()

    run_result = execute_python_code(code)

    if run_result["success"]:
        return (
            f"[Coder Worker: Execution Succeeded]\n"
            f"Code Executed:\n```python\n{code}\n```\n"
            f"Output (stdout):\n{run_result['stdout'] or '(No stdout produced)'}"
        )
    else:
        return (
            f"[Coder Worker: Execution Failed]\n"
            f"Code Attempted:\n```python\n{code}\n```\n"
            f"Error (stderr):\n{run_result['stderr']}"
        )


if __name__ == "__main__":
    print("Testing Coder Worker directly...")
    res = run_coder_worker("Print the sum of squares from 1 to 10")
    print(res)