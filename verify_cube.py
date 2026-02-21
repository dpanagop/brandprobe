import pandas as pd
from brandprobe import Runner, BaseEngine
from brandprobe.probers import PERSONAS, TEST_CASES

class MockEngine(BaseEngine):
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        return f"Mock response. System: {len(system_prompt)}, User: {len(user_prompt)}"

class MockScorerEngine(BaseEngine):
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        # Mock sentiment score
        return "0.75"

class MockProberEngine(BaseEngine):
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        if "generate exactly" in system_prompt.lower() and "personas" in system_prompt.lower():
            return '{"Dynamic Persona 1": "You are dynamic persona 1.", "Dynamic Persona 2": "You are dynamic persona 2."}'
        elif "generate exactly" in system_prompt.lower() and "test case" in system_prompt.lower():
            return '{"Dynamic Case 1": "What do you think of {brand}?", "Dynamic Case 2": "Would you buy {brand}?"}'
        return "{}"

def main():
    targets = ["BrandA", "BrandB"]
    methodologies = ["Direct", "Adversarial", "Implicit"]
    
    # Test dynamic generation
    from brandprobe.probers import DynamicProberGenerator
    generator = DynamicProberGenerator(MockProberEngine())
    dynamic_personas = generator.generate_personas(2, "gamers")
    dynamic_cases = generator.generate_test_cases(2, "reliability")
    
    print("Dynamically Generated Personas:", dynamic_personas)
    print("Dynamically Generated Test Cases:", dynamic_cases)
    
    # Add dynamic items to static items
    test_cases = list(TEST_CASES.keys()) + list(dynamic_cases.keys())
    personas = list(PERSONAS.keys()) + list(dynamic_personas.keys())

    # We also need to add them to the global dicts in memory for the Runner to use them
    PERSONAS.update(dynamic_personas)
    TEST_CASES.update(dynamic_cases)

    engine = MockEngine()
    scorer_engine = MockScorerEngine()
    runner = Runner(engine, llm_scorer_engine=scorer_engine)

    print("Running 3D Cube...")
    df = runner.run_cube(targets, methodologies, test_cases, personas)

    expected_rows = len(targets) * len(methodologies) * len(test_cases) * len(personas)
    actual_rows = len(df)

    print(f"Targets({len(targets)}) * Methodologies({len(methodologies)}) * Test Cases({len(test_cases)}) * Personas({len(personas)})")
    print(f"Expected Rows: {expected_rows}")
    print(f"Actual Rows: {actual_rows}")

    assert expected_rows == actual_rows, "Mismatch in row count!"

    output_path = "cube_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")
    print("Verification Successful!")

if __name__ == "__main__":
    main()
