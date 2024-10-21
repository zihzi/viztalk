import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from lida import Manager, TextGenerationConfig, llm
from viz_generator import VizGenerator
from viz_executor import ChartExecutor
from PIL import Image
import io
import base64 
from poster_generator import create_pdf
# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Set page config
st.set_page_config(page_icon="analysis.png",layout="wide",page_title="Data2Poster")
st.title("📊 Data2Poster")

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("data/Movies.csv")
    datasets["Housing"] = pd.read_csv("data/Housing.csv")
    datasets["Cars"] = pd.read_csv("data/Cars.csv")
    datasets["Colleges"] = pd.read_csv("data/Colleges.csv")
    datasets["Customers & Products"] = pd.read_csv("data/Customers & Products.csv")
    datasets["Department Store"] = pd.read_csv("data/Department Store.csv")
    datasets["Energy Production"] = pd.read_csv("data/Energy Production.csv")
    st.session_state["datasets"] = datasets
else:
    # Use the list already loaded
    datasets = st.session_state["datasets"]
# Set left sidebar content
with st.sidebar:
    # Set area for user guide
    with st.expander("UserGuide"):
         st.write("""
            1. Input your OpenAI Key.
            2. Select dataset from the list below or upload your own dataset.
        """)
    # Set area for OpenAI key
    openai_key = st.text_input(label = "🔑 OpenAI Key:", help="Required for models.",type="password")
         
    # First we want to choose the dataset, but we will fill it with choices once we"ve loaded one
    dataset_container = st.empty()

    # Upload a dataset(!can only use the latest uploaded dataset for now)
    try:
        uploaded_file = st.file_uploader("📂 Load a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets.
            file_name = uploaded_file.name[:-4]
            datasets[file_name] = pd.read_csv(uploaded_file)
            # default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio("👉 Choose your data:",datasets.keys(),index=index_no)
    # Save dimensions of the dataset for feature selection
    head = datasets[chosen_dataset].columns


# At the beginning, provide visualization about default dataset(Movies)
with st.chat_message("assistant"):
    st.write("Let's explore your data!✨")
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
        textgen_config_vis = TextGenerationConfig(n=5, temperature=0.5, model="gpt-4o-mini", use_cache=True)
        summary = lida.summarize(datasets[chosen_dataset],summary_method="llm",textgen_config=textgen_config) 

        # Call gpt-4o-mini to select a topic to analyze and related features in JSON format
        class FeatureSelector(BaseModel):
            Topic: str = Field(description="It is what you want to analyze given a multi-dimensional tabluar data.")
            Feature: str = Field(description="A list of dimension name in a multi-dimensional tabluar data that influence or related to the topic.")
        parser_1 = JsonOutputParser(pydantic_object = FeatureSelector)
        prompt = PromptTemplate(
            template="""[EXAMPLE] 
                            Consider a multi-dimensional tabluar data in python, and it contains about house sales in California in 2019.
                            There are dimensions such as: City, Month, House Style, Location, and so on...
                            If you are a real estate agent who wants to analyze what feature will influence the house price, you may consider location, house style, as potential features.                          
                        [/EXAMPLE]
                            You are an expert iny any domain.
                            Baed on data summary {summary}, think step by step about what topic to analyze and choose some potential features as list from {head}.\n{format_instructions}""",
            input_variables=["summary"],
            partial_variables={"format_instructions": parser_1.get_format_instructions()},
            
        )
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key = openai_key)
        chain = prompt | llm | parser_1
        topic_feature = chain.invoke(input={"summary":summary, "head":head})
        topic = topic_feature['Topic']
        features = topic_feature['Feature']
        # Call gpt-4o-mini to generate data scope in JSON format
        class DataScope(BaseModel):
            Subspace: str = Field(description="It refers to any categorical dimension in multi-dimensional tabluar data.")
            Breakdown: str = Field(description="ONLY dimension name whose attribute is CATEGORICAL.")
            Measure: str = Field(description="A aggregate functions in python,which is .count().It use to count different attribute in one subspace.")

        parser_2 = JsonOutputParser(pydantic_object=DataScope)
        prompt = PromptTemplate(
            template="""[EXAMPLE]A DataScope Example:
                            Consider a multi-dimensional tabluar data in python, and it contains about house sales in California in 2019.
                            There are dimensions such as: City, Month, House Style, Sales, and so on....
                            A pair of data scope can be constructed by 2 data scopes, e.g.,["Subspace": "data["City"] = Los Angeles", "Breakdown": "Month", "Measure": "data["Sales"].mean()"] 
                            and ["Subspace": "data["City"] = Fresno", "Breakdown": "Month", "Measure": "data["Sales"].mean()"].This pair of data scope can contrast the monthly average sales of houses in different cities.
                        [/EXAMPLE]
                            You are a professional data scientist who can explore data logically.
                            Given the topic {topic}, please refer to the {summary}, and choose features from {features} to generate data scopes pair that caninterpret the topic.
                            ONLY provide ten JSON object that can represents data scopes in one pairs.
                            In each pair, 2 data scopes can be contrasted by ONLY changing the subsapce.\n{format_instructions}""",
            input_variables=["summary"],
            partial_variables={"format_instructions": parser_2.get_format_instructions()},
        )
        chain = prompt | llm | parser_2
        data_scope_json = chain.invoke(input={"topic":topic, "summary":summary, "features":features})

        # Convert data_scope_json to a pandas DataFrame
        df_data_scopes = pd.DataFrame(data_scope_json)
     

        for i in range(0, len(df_data_scopes), 2):
            Subspace_1 = df_data_scopes["Subspace"][i]
            Breakdown_1 = df_data_scopes["Breakdown"][i]
            Measure_1 = df_data_scopes["Measure"][i]

            Subspace_2 = df_data_scopes["Subspace"][i+1]
            Breakdown_2 = df_data_scopes["Breakdown"][i+1]
            Measure_2 = df_data_scopes["Measure"][i+1]

            df_data_scope1 = [Subspace_1, Breakdown_1, Measure_1]
            df_data_scope2 = [Subspace_2, Breakdown_2, Measure_2]

            # Generate visualization code
            viz_generator = VizGenerator()
            response = viz_generator.generate(summary, df_data_scope1, df_data_scope2, textgen_config_vis, lida.text_gen)
    
            # Execute visualization code
            viz_executor = ChartExecutor()
            chart = viz_executor.execute(response[0], datasets[chosen_dataset], summary)
            
            
            imgdata = base64.b64decode(chart[0].raster)
            img = Image.open(io.BytesIO(imgdata))
                # st.write(f"Subspace:",df_data_scope1[0],
                #         "\n\nBreakdown:",df_data_scope1[1],
                #         "\n\nMeasure:",df_data_scope1[2])
                # st.write(f"Subspace:",df_data_scope2[0],
                #         "\n\nBreakdown:",df_data_scope2[1],
                #         "\n\nMeasure:",df_data_scope2[2])
            st.image(img, use_column_width=True)
            # image for pdf
            for i in range(0,2):
                img.save(f"image_{i}.png")
        # Create pdf and download
        create_pdf(chosen_dataset,openai_key) 
        st.success(f"""Poster has been created successfully!🎉""")
        with open(f"""{chosen_dataset}_summary.pdf""", "rb") as f:
            st.download_button("Download Poster", f, f"""{chosen_dataset}_summary.pdf""")

# Display chosen datasets 
if chosen_dataset :
   st.subheader(chosen_dataset)
   st.dataframe(datasets[chosen_dataset],hide_index=True)

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
