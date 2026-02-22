import pandas as pd
from typing import List, Optional
from .engines import BaseEngine
from .probers import PERSONAS, TEST_CASES
from .scorers import SentimentWrapper, ReliabilityLayer

class Runner:
    def __init__(self, engine: BaseEngine, llm_scorer_engine: Optional[BaseEngine] = None):
        self.engine = engine
        self.llm_scorer_engine = llm_scorer_engine

    def _apply_methodology(self, base_persona: str, methodology: str) -> str:
        """Modifies the base persona based on the methodology."""
        if methodology == "Direct":
            return f"{base_persona} Answer directly and honestly."
        elif methodology == "Adversarial":
            return f"{base_persona} Be highly critical and look for the worst possible interpretation."
        elif methodology == "Implicit":
            return f"{base_persona} Respond by telling a short personal story or narrative."
        return base_persona

    def run_cube(self, targets: List[str], methodologies: List[str], test_cases: List[str], personas: List[str], temperature: float = 0.7) -> pd.DataFrame:
        """
        Executes the stateless 3D Cube evaluation for target x methodology x test case x persona.
        Returns a Pandas DataFrame with the results.
        """
        results = []
        for target in targets:
            for methodology in methodologies:
                for test_case in test_cases:
                    for persona in personas:
                        if persona not in PERSONAS:
                            raise ValueError(f"Unknown Persona: {persona}")
                        if test_case not in TEST_CASES:
                            raise ValueError(f"Unknown Test Case: {test_case}")
                        
                        system_prompt = self._apply_methodology(PERSONAS[persona], methodology)
                        user_prompt = TEST_CASES[test_case].format(brand=target)
                        
                        # Stateless engine call
                        response = self.engine.generate(system_prompt, user_prompt, temperature=temperature)
                        
                        # Scorer processing
                        sentiment = SentimentWrapper.analyze(response, engine=self.llm_scorer_engine)
                        is_consistent = ReliabilityLayer.check_consistency(response)
                        
                        results.append({
                            "Target": target,
                            "Methodology": methodology,
                            "Test Case": test_case,
                            "Persona": persona,
                            "System Prompt": system_prompt,
                            "User Prompt": user_prompt,
                            "Response": response,
                            "Sentiment": sentiment,
                            "Is Consistent": is_consistent
                        })
                        
        return pd.DataFrame(results)
