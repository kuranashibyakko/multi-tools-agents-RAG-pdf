import os
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(timeout=20.0)
MODEL_NAME = "openai/gpt-oss-20b"

SANDBOX_DIR = Path("./sandbox").resolve()
SANDBOX_DIR.mkdir(exist_ok=True)


def extract_python_code(text: str) -> str:
    """Extracts Python code from markdown code fences or raw response."""
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def run_coder_worker(task: str) -> str:
    """Coder Persona: Generates Python code, executes it in an isolated
    subprocess sandbox, and supports headless matplotlib chart generation.
    """
    system_prompt = (
        "You are an expert Python Data Science & Simulation Engineer in a multi-agent network.\n"
        "Write a standalone, self-contained Python script to solve the task.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY valid Python code inside a ```python ... ``` block.\n"
        "2. Always print() numerical summary results clearly to stdout.\n"
        "3. PLOTTING & CHARTS: If the task asks for a plot, histogram, chart, or visual comparison, "
        "use `import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt` and "
        "save the figure strictly as `'output_plot.png'` using "
        "`plt.savefig('output_plot.png', dpi=150, bbox_inches='tight')`. Never call `plt.show()`.\n"
        "4. Do not use external APIs or input() prompts."
    )

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write and execute Python code for: {task}"},
        ],
        temperature=0.0,
    )

    raw_output = response.choices[0].message.content or ""
    clean_code = extract_python_code(raw_output)

    script_path = SANDBOX_DIR / "temp_script.py"
    plot_path = SANDBOX_DIR / "output_plot.png"

    if plot_path.exists():
        plot_path.unlink()

    script_path.write_text(clean_code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SANDBOX_DIR),
            capture_output=True,
            text=True,
            timeout=15,
        )

        stdout_str = result.stdout.strip() or "(No stdout printed)"
        stderr_str = result.stderr.strip()

        if result.returncode == 0:
            plot_notice = ""
            if plot_path.exists():
                plot_notice = "[ARTIFACT CREATED]: Chart saved to sandbox/output_plot.png\n"

            # Put stdout and artifact confirmation FIRST so they are never truncated
            return (
                f"[Coder Worker: Execution Succeeded]\n"
                f"{plot_notice}"
                f"Standard Output:\n```\n{stdout_str}\n```\n"
                f"Code Executed:\n```python\n{clean_code}\n```"
            )
        else:
            return (
                f"[Coder Worker: Execution Failed (Exit Code {result.returncode})]\n"
                f"Error Traceback:\n```\n{stderr_str}\n```\n"
                f"Code Attempted:\n```python\n{clean_code}\n```"
            )

    except subprocess.TimeoutExpired:
        return "[Coder Worker: Execution Failed] Script timed out after 15 seconds."
    except Exception as e:
        return f"[Coder Worker: Execution Error] {str(e)}"