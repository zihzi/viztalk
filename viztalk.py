import pandas as pd
# from openai import OpenAI
import streamlit as st



# import warnings
# warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import conversational_retrieval
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS


st.set_page_config(page_icon="analysis.png",layout="wide",page_title="VizTalk")
# Set page content
st.title("📈VizTalk🔍")
# st.subheader("Give you insightful visualization through conversation!")

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

# Set left sidebar content
with st.sidebar:
    # Set area for user guide
    with st.expander("UserGuide"):
         st.write("""
            1. Input your OpenAI Key.
            2. Select dataset from the list below or upload your own dataset.
            3. Type your question in the text area.
        """)
#Set area for OpenAI key
    openai_key = st.text_input(label = ":key: OpenAI Key:", help="Required for models.",type="password")
         
# First we want to choose the dataset, but we will fill it with choices once we"ve loaded one
    dataset_container = st.empty()

    # Upload a dataset
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

# Initalize the chatbot
with st.chat_message("assistant"):
    st.write("Give you insightful visualization through conversation!✨")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Execute chatbot query
    # Text input for query
if question := st.chat_input("Talk your data!"):
    api_keys_entered = True
    # Check API keys are entered.
    if not openai_key.startswith("sk-"):
        st.error("Please enter a valid OpenAI API key.")
        api_keys_entered = False
    if api_keys_entered:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

            # try:
            templates = load_templates()
            prompt_template_1, prompt_template_2, chat_template = templates

            prompt_1 = PromptTemplate.from_template(template = prompt_template_1)
            llm_1 = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key = openai_key)
            chain_1 = prompt_1 | llm_1 | StrOutputParser()

            prompt_2 = PromptTemplate.from_template(template=prompt_template_2)
            llm_2 = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key = openai_key)#,model_kwargs={"stop": ["final table:"]})
            chain_2 =  {"chosen_dataset": lambda x:datasets[chosen_dataset], "enhance_query": RunnablePassthrough()} | prompt_2 | llm_2.bind(stop = ["final table:"])| CommaSeparatedListOutputParser()
                
            link_chain = ({"chain1":chain_1}
                        | RunnablePassthrough.assign(chain2=chain_2))
            materials = link_chain.invoke({"head":head,"question":question})

            # chat_prompt = ChatPromptTemplate.from_template(template=chat_template)
            prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                        "system",
                        """You are a helpful assistant who's good at data visualization.Base on {query} and {table}, only provide code script start with code  "import ...".""",
                        ),MessagesPlaceholder(variable_name="chat_history"), ("human","{query}")
                    ]
)

            chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key = openai_key)
            chat_chain = prompt | chat_llm 
            # {"query":RunnablePassthrough(), "table":RunnablePassthrough()}|
            # | StrOutputParser()
            store={}
            def get_session_history(session_id):
                if session_id not in store:
                    store[session_id]= ChatMessageHistory()
                return store[session_id]
            # chat_history_for_chain = ChatMessageHistory()

            chain_with_message_history = RunnableWithMessageHistory(
                            chat_chain,
                            get_session_history,
                            input_messages_key="query",
                            # output_messages_key="output_messages",
                            history_messages_key="chat_history",
                            )
            answer = chain_with_message_history.invoke({"query":materials['chain1'],"table":materials['chain2']},config={"configurable":{"session_id":"1"}})
            st.session_state.messages.append({"role": "assistant", "content": answer})  
        with st.chat_message("assistant"):
            st.write(answer.content)
            plot_area = st.empty()
            plot_area.pyplot(exec(answer.content))

                
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
# Display plots
plots = st.columns(5)


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
