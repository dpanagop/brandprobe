# BrandProbe: Technical Walkthrough

BrandProbe is a modular framework designed for auditing LLM sentiment across multiple dimensions (Targets, Methodologies, Test Cases, and Personas) using a "3D Cube" approach. This document provides a technical walkthrough of the codebase.

## 1. Architectural Overview

The framework is built around decoupled components located in the `brandprobe/` package:

### 1.1 `engines.py` (The Communication Layer)
This module defines how BrandProbe communicates with AI models.
- **`BaseEngine`**: An abstract base class requiring a `generate(system_prompt, user_prompt, max_tokens, temperature)` method. This enforces a standard interface.
- **`AzureOpenAIEngine` / `OpenAIEngine`**: Concrete implementations that wrap specific SDKs (e.g., the official `openai` Python package). They handle authentication, endpoint routing, and parsing the API responses back into standard strings.

### 1.2 `probers.py` (The Templating & Generation Layer)
This module manages the inputs sent to the LLM.
- **Static Dictionaries**: Contains predefined `PERSONAS` (system prompts) and `TEST_CASES` (user prompts with a `{brand}` injection placeholder).
- **`DynamicProberGenerator`**: A utility class that accepts a `BaseEngine` to dynamically ask an LLM to generate new, JSON-formatted personas and test cases. The generation methods (`generate_personas`, `generate_test_cases`) now accept an optional `temperature` floating-point value to control the creativity of generated items. It includes static methods (`save_to_json`, `load_from_json`) for disk persistence.

### 1.3 `scorers.py` (The Evaluation Layer)
This module processes the LLM's raw text response.
- **`SentimentWrapper`**: Contains an `analyze(text, engine=None)` method. If no engine is provided, it defaults to a deterministic NLP approach using `TextBlob`. If an engine is provided, it uses the LLM as a "judge" to extract a sentiment score between -1.0 and 1.0.
- **`ReliabilityLayer`**: Provides basic heuristics (like `check_consistency`) to ensure the response isn't empty or malformed before scoring.

### 1.4 `runner.py` (The Orchestrator)
The `Runner` orchestrates the execution of the 3D Cube.
- **`__init__(engine, llm_scorer_engine=None)`**: Takes the primary generation engine and an optional secondary engine for scoring.
- **`run_cube(...)`**: Executes a nested loop across Targets, Methodologies, Test Cases, and Personas. You can pass an optional `temperature` (default `0.7`) to control generation determinism across the main audit run. For each combination:
  1. It formats the system and user prompts (`_apply_methodology`).
  2. Calls `self.engine.generate()` statelessly (no conversation history).
  3. Passes the result to `SentimentWrapper` and `ReliabilityLayer`.
  4. Appends the metrics into a list and returns a structural `pandas.DataFrame`.

## 2. Understanding `verify_cube.py`

The `verify_cube.py` script serves as an integration test to ensure the mathematical Cartesian product logic works without spending real API credits.

**How it works:**
1. **Mock Engines**: It defines a `MockEngine` (for generation), a `MockScorerEngine` (for scoring), and a `MockProberEngine` (for generating dynamic personas/cases). These classes inherit from `BaseEngine` but return hardcoded strings instead of making network calls.
2. **Dynamic Generation Test**: It uses the `MockProberEngine` to "generate" 2 new personas and 2 new test cases, demonstrating how to merge dynamic items into the static `TEST_CASES` and `PERSONAS` dictionaries.
3. **Execution**: It initializes the `Runner` with the mock engines and runs the cube.
4. **Verification**: It calculates the expected rows: `Len(Targets) * Len(Methodologies) * Len(Test Cases) * Len(Personas)`. It asserts that the resulting DataFrame length matches this exact number (ensuring no combinations were skipped).
5. **Export**: It saves the results to `cube_results.csv` to prove the DataFrame structural integrity.

## 3. Extending BrandProbe: Adding New LLM Connections

Because of the `BaseEngine` abstraction, adding support for a new LLM provider (like Anthropic, Google Gemini, or a custom local server) is trivial.

### Example: Connecting to a Custom URL API
If you have an LLM hosted on a custom internal URL that accepts JSON POST requests, you just create a new class in `engines.py` (or directly in your notebook):

```python
import requests
from brandprobe import BaseEngine

class CustomRESTEngine(BaseEngine):
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        payload = {
            "system_message": system_prompt,
            "user_message": user_prompt,
            "maxOutputTokens": max_tokens,
            "temp": temperature
        }
        
        response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            # Parse the specific JSON structure your custom API returns
            return response.json().get("model_reply", "")
        else:
            print(f"Error: {response.status_code}")
            return ""
```

Once defined, you simply pass it to the orchestrator:
`runner = Runner(engine=CustomRESTEngine(url, key))`
