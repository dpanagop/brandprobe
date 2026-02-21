# BrandProbe: Detailed User Guide

Welcome to the comprehensive user guide for BrandProbe. This document will walk you through setting up the framework, configuring models for various roles (testing, scoring, generating), and extending execution.

## 1. Setting Up the Models

In BrandProbe, models are treated as "Engines". You can use the same engine for everything, or mix and match specialized models for different tasks.

### 1.1 The Generation Engine (The "Test Subject")
This is the model you want to audit.

```python
from brandprobe import OpenAIEngine, AzureOpenAIEngine

# Example 1: Standard OpenAI (e.g. GPT-4o)
generation_engine = OpenAIEngine(api_key="sk-...", model="gpt-4o")

# Example 2: Local Ollama Model
# Assume Ollama is running locally on port 11434
generation_engine = OpenAIEngine(api_key="ollama", base_url="http://localhost:11434/v1", model="llama3")

# Example 3: Azure OpenAI
azure_engine = AzureOpenAIEngine(
    api_key="your-azure-key", 
    api_version="2024-02-15-preview", 
    azure_endpoint="https://your-resource.openai.azure.com/", 
    deployment_name="gpt-4-deployment"
)
```

### 1.2 The Scoring / Judging Engine (Optional)
If you want an LLM to calculate the final sentiment score instead of the default deterministic NLP (`TextBlob`), you instantiate a second engine. This is usually a highly capable model designed for logical reasoning or extraction.

```python
# Create a dedicated scorer model
scoring_engine = OpenAIEngine(api_key="sk-...", model="gpt-4-turbo")
```

### 1.3 The Prompter Engine (Optional)
If you want to instruct an LLM to dynamically invent new Personas and Test Cases, you can instantiate a third engine (or reuse an existing one).

```python
# We'll just reuse the generation_engine for this guide
prompter_engine = generation_engine 
```

---

## 2. Choosing How Sentiment is Calculated

When configuring the Runner orchestration, you decide how sentiment is scored by whether or not you provide a `llm_scorer_engine`.

**Option A: Deterministic NLP (TextBlob - Default)**
If you do not pass a scoring engine, BrandProbe defaults to `TextBlob`. This is faster, cheaper, and strictly deterministic based on word polarity.
```python
from brandprobe import Runner
runner = Runner(engine=generation_engine)
```

**Option B: LLM as a Judge**
If you pass the scoring engine, BrandProbe asks the LLM to read the generated response and assign it a polarity score from `-1.0` (negative) to `1.0` (positive).
```python
runner = Runner(engine=generation_engine, llm_scorer_engine=scoring_engine)
```

---

## 3. Creating, Saving, and Loading Personas & Test Cases

BrandProbe comes with default personas and test cases, but its real power is dynamic generation using the `DynamicProberGenerator`.

### 3.1 Generating Dynamics Prompts
```python
from brandprobe import DynamicProberGenerator
from brandprobe.probers import PERSONAS, TEST_CASES

generator = DynamicProberGenerator(prompter_engine)

# Tell the LLM to invent 3 personas representing 'European Teenagers'
new_personas = generator.generate_personas(count=3, context="European Teenagers")

# Tell the LLM to invent 4 test cases regarding 'Data Privacy'
new_cases = generator.generate_test_cases(count=4, topic="Data Privacy")

# Add the newly generated items to the global dictionaries in memory so the Runner uses them
PERSONAS.update(new_personas)
TEST_CASES.update(new_cases)
```

### 3.2 Saving to Disk
To avoid repeating expensive LLM generation calls, save your new prompts to JSON.
```python
# Saves dictionaries to disk. If the file exists, it appends/updates it safely.
DynamicProberGenerator.save_to_json(new_personas, 'custom_personas.json')
DynamicProberGenerator.save_to_json(new_cases, 'custom_cases.json')
```

### 3.3 Loading from Disk
In a future Jupyter Notebook session, you can skip generation and just load from disk.
```python
loaded_personas = DynamicProberGenerator.load_from_json('custom_personas.json')
PERSONAS.update(loaded_personas)
```

---

## 4. Running the Complete Audit

Once your engines are initialized and your dictionaries are populated, you execute the 3D Cube.

```python
# 1. Define your Targets and Methodologies
targets = ["Apple", "Google"]
methodologies = ["Direct", "Adversarial", "Implicit"]

# 2. Extract keys from the dictionaries
target_test_cases = list(TEST_CASES.keys())
target_personas = list(PERSONAS.keys())

# 3. Create Runner instance
runner = Runner(engine=generation_engine, llm_scorer_engine=scoring_engine)

# 4. Execute the framework
df = runner.run_cube(targets, methodologies, target_test_cases, target_personas)

# 5. Export and Analyze
df.to_csv("apple_google_sentiment_audit.csv", index=False)
print(df.head())
```

### Tips for Execution
- **Methodologies**: "Direct" answers plainly. "Adversarial" asks the persona to assume the worst. "Implicit" asks the persona to respond creatively via a narrative.
- **Statelessness**: Every row generated during `run_cube` is strictly stateless. No conversational context is retained between prompts.
- **Data Analysis**: Because the output is a standard pandas DataFrame (`df`), you can use standard Jupyter Notebook workflows (matplotlib, seaborn, groupby) to map sentiment heatmaps by Persona vs Target immediately after generation. 
