import pandas as pd
import openai
import streamlit as st
#import streamlit_nested_layout
# from classes import get_primer,format_question,run_request
# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
# 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

st.set_page_config(page_icon="analysis.png",layout="wide",page_title="VizTalk")
# Set page content
st.title("📈VizTalk🔍")
st.subheader("Give you insightful visualization through conversation!")

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("movies.csv")
    datasets["Housing"] =pd.read_csv("housing.csv")
    datasets["Cars"] =pd.read_csv("cars.csv")
    datasets["Colleges"] =pd.read_csv("colleges.csv")
    datasets["Customers & Products"] =pd.read_csv("customers_and_products_contacts.csv")
    datasets["Department Store"] =pd.read_csv("department_store.csv")
    datasets["Energy Production"] =pd.read_csv("energy_production.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

#Set area for OpenAI key
openai_key = st.text_input(label = ":key: OpenAI Key:", help="Required for models.",type="password")

# Set left sidebar content
with st.sidebar:
    # Set area for user guide
    with st.expander("UserGuide"):
         st.write("""
            1. Input your OpenAI Key.
            2. Select dataset from the list below or upload your own dataset.
            3. Type your question in the text area.
        """)
         
# First we want to choose the dataset, but we will fill it with choices once we"ve loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    try:
        uploaded_file = st.file_uploader("📂 Load a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets. Give it a nice name.
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            # default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio("📊 Choose your data:",datasets.keys(),index=index_no)
    # pass to prompt template
    head = list(datasets[chosen_dataset].columns)

# load template
def load_templates():

    templates = []
    for filename in ["prompt_template_1", "prompt_template_2", "chat_template"]:
        with open(f"./templates/{filename}.txt", "r") as file:
            templates.append(file.read())

    return templates

# Execute chatbot query
with st.form("my_form"):
     # Text area for query
    question = st.text_area("Talk your data!",height=10)
    submitted = st.form_submit_button("Submit")
    if submitted > 0:
        api_keys_entered = True
        # Check API keys are entered.
        if not openai_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
        if api_keys_entered:
            # Place for plots depending on how many models
            # plots = st.columns(1) 
            # Create model, run the request and print the results
            # with plots:
            # try:
                templates = load_templates()
                prompt_template_1, prompt_template_2, chat_template = templates

                prompt_1 = PromptTemplate.from_template(template = prompt_template_1)
                llm_1 = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key = openai_key)
                chain_1 = prompt_1 | llm_1 | StrOutputParser()

                prompt_2 = PromptTemplate.from_template(template=prompt_template_2)
                llm_2 = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key = openai_key)#,model_kwargs={"stop": ["final table:"]})
                chain_2 =  {"chosen_dataset": lambda x:datasets[chosen_dataset], "enhance_query": RunnablePassthrough()} | prompt_2 | llm_2.bind(stop = ["final table:"])| CommaSeparatedListOutputParser()

                chat_prompt = ChatPromptTemplate.from_template(template=chat_template)
                chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key = openai_key)
                chat_chain = {"table":RunnablePassthrough(),"query":RunnablePassthrough()} | chat_prompt | chat_llm | StrOutputParser()

                final_chain = ({"chain1":chain_1}
                            | RunnablePassthrough.assign(chain2=chain_2)
                            | RunnablePassthrough.assign(chatchain=chat_chain))
                answer = final_chain.invoke({"head":head,"question":question})
                #######fail to use pd_dataframe_agent########
                # df = datasets[chosen_dataset]
                # llm_agent = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key = openai_key)
                # agent_executor = create_pandas_dataframe_agent( llm_agent,df,input_variables = answer["chain1"],agent_type= "zero-shot-react-description" , verbose= True , return_intermediate_steps= True)
                # prompt_agent = '''Based on the {query}, generate correct df which can visualize the data that meet the query.Just provide the final df.'''
                # agent_executor.invoke(prompt_agent)
                # print(answer["chatchain"])
                ############################################
                st.info(answer)
                plot_area = st.empty()
                plot_area.pyplot(exec(answer["chatchain"]))        
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

# Display the datasets in a list of tabs
# Create the tabs
tab_list = st.tabs(datasets.keys())

# Load up each tab with a dataset
for dataset_num, tab in enumerate(tab_list):
    with tab:
        # Can"t get the name of the tab! Can"t index key list. So convert to list and index
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name],hide_index=True)

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
