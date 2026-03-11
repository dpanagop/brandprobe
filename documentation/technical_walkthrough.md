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
- **`SentimentWrapper`**: Contains an `analyze(text, engine=None, method="textblob")` method. It defines the Sentiment Tier System:
  - **Tier 1 (`method="textblob"`)**: Deterministic NLP approach using `TextBlob`. (Fastest, Local, Lexicon-based)
  - **Tier 2 (`method="roberta"`)**: Loads a local HuggingFace Pipeline `cardiffnlp/twitter-roberta-base-sentiment-latest` via transformers. It caches the model lazily. (Recommended for local high-accuracy)
  - **Tier 3 (`method="llm"`)**: Uses the LLM as a "judge" (requires passing an `engine`) to extract a sentiment score between -1.0 and 1.0. (Highest nuance, API-dependent)
- **`ReliabilityLayer`**: Provides basic heuristics (like `check_consistency`) to ensure the response isn't empty or malformed before scoring.

### 1.4 `runner.py` (The Orchestrator)
The `Runner` orchestrates the execution of the 3D Cube.
- **`__init__(engine, llm_scorer_engine=None, sentiment_method="textblob")`**: Takes the primary generation engine, an optional secondary engine for scoring, and the preferred sentiment tier.
- **`run_cube(...)`**: Executes a nested loop across Targets, Methodologies, Test Cases, and Personas. You can pass an optional `temperature` (default `0.7`) to control generation determinism across the main audit run. For each combination:
  1. It formats the system and user prompts (`_apply_methodology`).
  2. Calls `self.engine.generate()` statelessly (no conversation history).
  3. Passes the result to `SentimentWrapper` and `ReliabilityLayer`.
  4. Appends the metrics into a list and returns a structural `pandas.DataFrame`.

### 1.5 `analytics.py` (The Semantic Auditing Layer)
This module introduces automated semantic analysis using local embeddings.
- **`get_consistency_metrics(df, group_cols)`**: Utilizes `SentenceTransformer` locally (`all-MiniLM-L6-v2`) to generate embeddings for valid results in the DataFrame. It calculates pairwise Cosine Similarity among grouped responses to identify semantic consistency. Higher values mean the LLM is responding similarly across attempts.
- **`calculate_sentiment_skew(df, target_col='Sentiment', group_cols=None)`**: Uses `scipy.stats.skew` on grouped DataFrame columns to calculate the multi-dimensional Fisher-Pearson skewness of sentiment scores. It assigns text labels to let you know if a model cut leans overly positive or overly negative.

### 1.6 `visualizations.py` (The Charting Layer)
This module provides dynamic charting for presentation and qualitative review.
- **`plot_radar(df, target_names, axis_col, score='Sentiment', target_variable='Target', ...)`**: Generates a dynamic radar chart using `matplotlib` polar projection, smoothly comparing selected Targets across dynamic categorical axes (e.g., Personas or Methodologies). Optimized for sentiment ranges from -1 to 1 (visualized from -1.1 to 1.1).
- **`plot_semantic_map(df, target_filter, color_by, target_variable='Target', ...)`**: Generates a 2D map. It generates local `all-MiniLM-L6-v2` embeddings, squashes them down to 2 dimensions using `umap-learn`, and plots a scatter graph with `seaborn` colored by a desired parameter.
- **`plot_skew_comparison(skew_df, x_col, hue_col, ...)`**: Generates a Bar Chart displaying the calculated Skewness for different groupings (e.g., comparing Targets side-by-side grouped by Persona) complete with analytical threshold lines at 0.5 and -0.5.


## 2. Testing Scripts and Verification

The repository contains several scripts designed to verify logical integrity, visualization outputs, and sentiment pipelines without spending real API credits or requiring live models.

### 2.1 `verify_cube.py` (Integration Testing)
Serves as an integration test to ensure the mathematical Cartesian product logic works.
- **Mock Engines**: It defines a `MockEngine` (for generation), a `MockScorerEngine` (for scoring), and a `MockProberEngine` (for generating dynamic personas/cases). These inherit from `BaseEngine` but return hardcoded strings.
- **Dynamic Generation Test**: Uses the `MockProberEngine` to "generate" new items, demonstrating merging into the static `TEST_CASES` and `PERSONAS`.
- **Execution & Verification**: Initializes the `Runner` with mock engines, runs the cube, and asserts the resulting DataFrame length matches the expected mathematical rows (`Len(Targets) * Len(Methodologies) * Len(Test Cases) * Len(Personas)`).
- **Export**: Saves results to `cube_results.csv` to prove structural integrity.

### 2.2 `test_visuals.py` (Visualization & Analytics Testing)
Validates the local `SentenceTransformer` caching and charting functions.
- **Synthetic Data**: Generates a dummy pandas DataFrame mirroring the output of a real `run_cube` execution (with mock targets, personas, and sentiment scores).
- **Analytics Check**: Runs the synthetic DataFrame through `get_consistency_metrics` and `calculate_sentiment_skew` to verify local embedding performance and dimensional grouping calculations.
- **Chart Generation**: Generates sample UMAP representations, Radar Charts, and Skew Bar Plots, saving them locally to a `test_output` directory to ensure mathematical errors (like NaN or division by zero handling) do not disrupt visual generation.

### 2.3 `test_roberta.py` (Sentiment Tier 2 Testing)
A micro-script to validate the HuggingFace `transformers` integration.
- Passes explicit strings (Positive, Negative, Neutral) through the `SentimentWrapper`.
- Verifies that the internal logic correctly maps the model-specific textual labels outputted by `cardiffnlp/twitter-roberta-base-sentiment-latest` into continuous mathematical floats between `-1.0` and `1.0`.

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
