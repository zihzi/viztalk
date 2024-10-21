import streamlit as st
import pandas as pd
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
from PIL import Image
import io
import base64 
# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)


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
urls=["data/Movies.csv","data/Housing.csv","data/Cars.csv","data/Colleges.csv","data/Customers & Products.csv","data/Department Store.csv","data/Energy Production.csv"]
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
            uploaded_file_path = os.path.join(uploaded_file.name)
            datasets[file_name].to_csv(uploaded_file_path, index=False)
            urls.append(uploaded_file_path)
            # default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio("👉 Choose your data:",datasets.keys(),index=index_no)
    # Get the dataset for lida
    for url in urls: 
        if chosen_dataset == url[5:-4]:
           lida_dataset = url

# At the beginning, provide visualization about default dataset(Movies)
with st.chat_message("assistant"):
    st.write("Let me help you explore the data!✨")
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
        textgen_config_vis = TextGenerationConfig(n=1, temperature=0.2, model="gpt-4o-mini", use_cache=True)
        summary = lida.summarize(lida_dataset,summary_method="llm",textgen_config=textgen_config) 
        goals = lida.goals(summary, n=3, textgen_config=textgen_config)
        goal_questions = [goal.question for goal in goals] 
        goal_rationale = [goal.rationale for goal in goals]
        AImessage_to_append = f"""
                    Based on the data, you may be interested...
                    1. {goal_questions[0]}\n
                        {goal_rationale[0]}\n
                    2. {goal_questions[1]}\n
                        {goal_rationale[1]}\n
                    3. {goal_questions[2]}\n
                         {goal_rationale[2]}
                    """   
        with st.chat_message("assistant"):
            st.write(AImessage_to_append)        
            for i in range(len(goals)):
                visualizations = lida.visualize(
                    summary=summary,
                    goal=goals[i],
                    textgen_config=textgen_config_vis
                    )
                try:
                    if visualizations[0]:   
                        imgdata = base64.b64decode(visualizations[0].raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, use_column_width=True)
                except Exception as e:
                    st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")
    

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])                   
# User starts to type
if question := st.chat_input("Enter your query..."):
    if not openai_key:
        st.error("Please enter a valid OpenAI API key.")
        api_keys_entered = False
    if openai_key:
        st.session_state.messages.append({"role": "user","content": question})
        with st.chat_message("user"):
         st.markdown(question)
        lida = Manager(text_gen= llm("openai", api_key=openai_key))
        textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)
        textgen_config_vis = TextGenerationConfig(n=4, temperature=1.5, model="gpt-4o-mini", use_cache=True)
        summary = lida.summarize(lida_dataset,summary_method="llm",textgen_config=textgen_config)   
        visualizations = lida.visualize(
                    summary=summary,
                    goal=question,
                    textgen_config=textgen_config_vis
                    )
        try:
            if visualizations[0]:   
                        imgdata = base64.b64decode(visualizations[0].raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, use_column_width=True)
        except Exception as e:
                st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")
          

                
            # except Exception as e:
            #     if type(e) == openai.APIConnectionError:
            #                 st.error("OpenAI API Error. Please try again a short time later. (" + str(e) + ")")
            #     elif type(e) == openai.APITimeoutError:
            #                 st.error("OpenAI API Error. Your request timed out. Please try again a short time later. (" + str(e) + ")")
            #     elif type(e) == openai.RateLimitError:
            #                 st.error("OpenAI API Error. You have exceeded your assigned rate limit. (" + str(e) + ")")
            #     elif type(e) == openai.APIConnectionError:
            #                 st.error("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings. (" + str(e) + ")")
            #     elif type(e) == openai.BadRequestError:
            #                 st.error("OpenAI API Error. Your request was malformed or missing required parameters. (" + str(e) + ")")
            #     elif type(e) == openai.AuthenticationError:
            #                 st.error("Please enter a valid OpenAI API Key. (" + str(e) + ")")
            #     elif type(e) == openai.InternalServerError:
            #                 st.error("OpenAI Service is currently unavailable. Please try again a short time later. (" + str(e) + ")")               
            #     else:
            #                 st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

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
