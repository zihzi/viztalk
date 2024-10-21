import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from lida import Manager, TextGenerationConfig, llm
from viz_generator import VizGenerator
from viz_executor import ChartExecutor
import os
from PIL import Image
import io
import base64 
# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = "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"

# Set page config
st.set_page_config(page_icon="analysis.png",layout="wide",page_title="Data2Poster")
st.title("📊 Data2Poster")


# Set left sidebar content
with st.sidebar:
    # Set area for user guide
    with st.expander("UserGuide"):
         st.write("""
            1. Input your OpenAI Key.
            2. Select dataset from the list below or upload your own dataset.
            3. Type your question in the text area.
        """)
    # Set area for OpenAI key
    openai_key = st.text_input(label = "🔑 OpenAI Key:", help="Required for models.",type="password")
    lida_dataset = pd.read_csv("data/Movies.csv")

# At the beginning, provide visualization about default dataset(Movies)
with st.chat_message("assistant"):
    st.write("Let me to help you explore the data!✨")
    try_true = st.button("Try it out!")
if try_true:
    api_keys_entered = True
    # Check API keys are entered.
    if not openai_key.startswith("sk-"):
        st.error("Please enter a valid OpenAI API key.")
        api_keys_entered = False
    if api_keys_entered:
        lida = Manager(text_gen= llm("openai", api_key=openai_key))
        textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)
        # textgen_config_vis = TextGenerationConfig(n=5, temperature=0.5, model="gpt-4o-mini", use_cache=True)
        summary = lida.summarize(lida_dataset,summary_method="llm",textgen_config=textgen_config) 

        # Call gpt-4o-mini to generate data scope in JSON format
        class DataScope(BaseModel):
            Subspace: str = Field(description='A set of dimensions.It refers to any categorical column in tabluar data.')
            Breakdown: str = Field(description='ONLY columns name contain temporal words RESTRICTED to YEAR, MONTH, DAY, DATE.')
            Measure: str = Field(description='A set of aggregate functions in python such as .mean(), .min(), .max(), .sum(), .median().It use any numerical column in tabluar data.')

        parser = JsonOutputParser(pydantic_object=DataScope)
        prompt = PromptTemplate(
            template='''[EXAMPLE]A DataScope Example:
                            Consider a multi-dimensional tabluar data read as "df" in python, and it contains about house sales in California in 2019.
                            There are dimensions such as: City, Month, House Style, Sales, and so on....
                            A data scope can be constructed as ['Subspace': 'City', 'Breakdown': 'Month', 'Measure': 'data['Sales].mean()'] which means the average sales of houses in Los Angeles in each month.
                        [/EXAMPLE]
                            You are a professional data scientist who can explore data logically.
                            Please refer to the {summary} and think about how to explore this data, then use 'field_names' to provide five JSON object that represents data scopes which may contain interesting insight.\n{format_instructions}''',
            input_variables=["summary"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = ChatOpenAI(model_name='gpt-4o-mini')
        chain = prompt | llm | parser
        data_scope_json = chain.invoke(input={'summary':summary})
        # print(data_scope_json)

        # Convert data_scope_json to a pandas DataFrame
        df_data_scopes = pd.DataFrame(data_scope_json)

        for i in range(len(df_data_scopes)):
            Subspace = df_data_scopes['Subspace'][i]
            Breakdown = df_data_scopes['Breakdown'][i]
            Measure = df_data_scopes['Measure'][i]
            df_data_scope = [Subspace, Breakdown, Measure]

            # Generate visualization code
            viz_gen = VizGenerator()
            response = viz_gen.generate(summary,df_data_scope, textgen_config, lida.text_gen)
    
            # Execute visualization code
            viz_executor = ChartExecutor()
            chart = viz_executor.execute(response[0], lida_dataset, summary)
            
            imgdata = base64.b64decode(chart[0].raster)
            img = Image.open(io.BytesIO(imgdata))
            st.image(img, use_column_width=True)



# Insert footer to reference dataset origin  
footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style="display: block; text-align: center;"> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
