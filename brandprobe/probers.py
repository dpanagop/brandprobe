PERSONAS = {
    "Gen Z Activist": "You are a Gen Z activist passionate about social justice, climate change, and corporate responsibility. You speak colloquially.",
    "Frugal Retiree": "You are a frugal retiree on a fixed income, highly conscious of value and traditional expectations. You speak formally and carefully.",
    "Tech Optimist": "You are a tech optimist who loves gadgets, innovation, and believes technology solves all problems. You use tech jargon.",
    "Skeptical Journalist": "You are a skeptical journalist always looking for the hidden catch, greenwashing, or corporate spin. You are analytical and questioning.",
    "Corporate Executive": "You are a busy corporate executive focused on ROI, efficiency, and market trends. You communicate directly and professionally."
}

TEST_CASES = {
    "Recommendation": "Would you recommend {brand} to your friends or colleagues? Why or why not?",
    "Adjective Association": "What are the first three adjectives that come to mind when you hear {brand}? Explain briefly.",
    "Crisis Reflection": "If {brand} was involved in a major data breach, how would that affect your view of them?",
    "Value": "Do you feel that products from {brand} are worth the price?",
    "Future": "Where do you see {brand} in 5 years compared to its competitors?"
}

import json
from typing import Dict
from .engines import BaseEngine

class DynamicProberGenerator:
    def __init__(self, engine: BaseEngine):
        self.engine = engine

    def generate_personas(self, count: int, context: str = "general consumers") -> Dict[str, str]:
        """Generates a dictionary of Persona Name: Persona Prompt."""
        system_prompt = (
            "You are an expert market research persona designer. "
            f"Generate exactly {count} distinct personas representing {context}. "
            "Return the result ONLY as a valid JSON object where keys are the persona names (e.g. 'Gen Z Gamer') "
            "and values are their detailed behavioral system prompts (starting with 'You are a ...'). "
            "Do not include any Markdown formatting or comments in your response."
        )
        response = self.engine.generate(system_prompt, "Generate the personas.", max_tokens=1000)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM persona generation. Response was: {response}")
            return {}

    def generate_test_cases(self, count: int, topic: str = "general brand perception") -> Dict[str, str]:
        """Generates a dictionary of Test Case Name: Test Case Format String (must contain {brand})."""
        system_prompt = (
            "You are an expert market research questionnaire designer. "
            f"Generate exactly {count} distinct test case questions about {topic}. "
            "Each question MUST contain the placeholder '{brand}' where the brand name will be injected. "
            "Return the result ONLY as a valid JSON object where keys are short test case names (e.g. 'Quality Perception') "
            "and values are the question strings containing '{brand}'. "
            "Do not include any Markdown formatting or comments in your response."
        )
        response = self.engine.generate(system_prompt, "Generate the test cases.", max_tokens=1000)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse LLM test case generation. Response was: {response}")
            return {}

    @staticmethod
    def save_to_json(data: Dict[str, str], filepath: str):
        """Saves a dictionary to a JSON file, appending or overwriting it."""
        import os
        existing_data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Update existing records with the new data
        existing_data.update(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
            
    @staticmethod
    def load_from_json(filepath: str) -> Dict[str, str]:
        """Loads a dictionary of personas or test cases from a JSON file."""
        import os
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {}
