from dataclasses import asdict
from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from viz_scaffold import ChartScaffold


system_prompt = """
You are a helpful assistant highly skilled in writing PERFECT code for visualizations. 
Given some code template, you complete the template to generate a visualization given the dataset. 
The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. apply the right transformation, use the right visualization type, 
use the right data encoding, and use the right aesthetics (e.g., ENSURE x-label and y-label are both visible, ensure axis are legible). 
The transformations you apply MUST be correct and the fields you use MUST be correct. 
The visualization CODE MUST BE EXECUTABLE and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). 
You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply if you need to construct a new column, 
what fields you would use, what visualization type you would use, what aesthetics you would use, etc.
"""


class VizGenerator(object):
    """Generate visualizations from prompt"""

    def __init__(
        self
    ) -> None:

        self.scaffold = ChartScaffold()

    def generate(self, summary: Dict, df_data_scope: Dict, textgen_config: TextGenerationConfig, text_gen: TextGenerator):
        """Generate visualization code given a summary"""

        template, instructions = self.scaffold.get_template(df_data_scope)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary} \n\n"},
            {"role": "assistant", "content": f"I'll give you some rules to follow, and I need you to remember the rules, and write EXECUTABLE code that do not violate any rules.\n\n"},
            instructions,
            {"role": "user",
             "content":
             f"""
             The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing (field_names). 
             Only use variables that have been defined in the code or are in the dataset summary. 
             You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation.
             THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {template} \n\n.
             The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ..."""}]

        completions: TextGenerationResponse = text_gen.generate(
            messages = messages, config=textgen_config)
        response = [x['content'] for x in completions.text]

        return response