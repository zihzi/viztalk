from dataclasses import asdict
from typing import Dict


class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(
        self,
    ) -> None:

        pass

    def get_template(self, df_data_scope1: Dict, df_data_scope2: Dict):

        instructions = f"""
        Use {df_data_scope1[0]} and grouped by {df_data_scope1[1]} as x-axis,and use {df_data_scope1[2]} as y-axis to create a Bar Chart 1.
        Use {df_data_scope2[0]} and grouped by {df_data_scope2[1]} as x-axis,and use {df_data_scope2[2]} as y-axis to create a Bar Chart 2.
        Bar Chart 1 and Bar Chart 2 ALWAYS comply with the following rules: 
        1. Use 'fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))' to create a 1x2 grid of subplots.
        2. NEVER add a legend.
        3. ALWAYS make the y-axis value in percentage format, so normalize the y-axis value into percentage before drawing the bar.
        4. ALWAYS set the y-axis ticks to 0, 25, 50, 75, and 100 in the bar chart.
        5. ALWAYS make sure the x-axis and y-axis labels are visible (e.g., rotate when needed). 
        6. Use '.spines[:].set_visible(False)'to remove all spines.
        7. Use '.grid(axis = 'y', color="lightgray", linestyle="dashed", zorder=-10)' to add a horizontal grid line.
        8. Use '.title('', wrap=True, fontsize=8)' to add a title which is a question that can be answered by the subplot.
        9. Make sure to use 'color = '[#CDE8E5, #77B0AA, #135D66]''.Using 'edgecolor='none'' in EACH SUBPLOT.   
        Solve the task carefully by completing ONLY the <imports> AND <stub> section. DO NOT include plt.show().DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable data.
        The plot method must return a matplotlib object (plt). Think step by step.
        """
        # instruction to draw a map:"Use BaseMap for charts that require a map."

        
        template = \
                f"""
import matplotlib.pyplot as plt
import pandas as pd
<imports>
# plan -
def plot(data: pd.DataFrame):
    <stub> # only modify this section
    

    # Bar Chart 1 
     <stub> # only modify this section
   

    # Bar Chart 2 
     <stub> # only modify this section


    return plt;

chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""
        

      

        return template, instructions
    

        