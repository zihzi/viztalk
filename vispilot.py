import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lida import Manager, TextGenerationConfig, llm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from scipy.stats import wasserstein_distance_nd
data = pd.read_csv("clean_titanic.csv")
# data["pc_class"] = data["pc_class"].astype(str)
head = data.columns
openai_key = ""

lida = Manager(text_gen= llm("openai", api_key=openai_key))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)
summary = lida.summarize(data,summary_method="llm",textgen_config=textgen_config) 

# Call gpt-4o-mini to select a topic to analyze and related features in JSON format
class FeatureSelector(BaseModel):
    Topic: str = Field(description="It is a dimension name you want to analyze given a multi-dimensional tabluar data.")
    Feature: str = Field(description="A list of dimension name in a multi-dimensional tabluar data that influence or related to the topic.")
parser_1 = JsonOutputParser(pydantic_object = FeatureSelector)
prompt = PromptTemplate(
            template="""[EXAMPLE] 
                            Consider a multi-dimensional tabluar data in python, and it contains about house sales in California in 2019.
                            There are dimensions such as: City, Month, House Style, Location, house price and so on...
                            If you are a real estate agent who wants to analyze what feature will influence the house price(main feature), you may consider location, house style, as potential features.                          
                        [/EXAMPLE]
                            You are an expert iny any domain.
                            Baed on data summary {summary}, choose a dimension name from {head} to analyze.Choose some other potential features as list from {head}.\n{format_instructions}""",
            input_variables=["summary"],
            partial_variables={"format_instructions": parser_1.get_format_instructions()},
            
        )
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key = openai_key)
chain = prompt | llm | parser_1
topic_feature = chain.invoke(input={"summary":summary, "head":head})
topic = topic_feature["Topic"]
features = topic_feature["Feature"]

# Generate filters from gpt selected columns(topic and features)
def making_filter_list(data):
    filters = []
    for column in data.columns:
        if column in features or column == topic:
            values = data[column].unique()
            if len(values) > 5 and data[column].dtype in ["int64", "float64"]:
                median_value = round(data[column].mean(), 1)
                groups = ["<=" + str(median_value), ">" + str(median_value)]
                for group in groups:
                    filters.append(f"{column}{group}")
            else:
                for value in values:
                    filters.append(f"{column}={value}")
    return filters 
filters = making_filter_list(data)

# Filter out data with as overall(enumerate each filter in filters)
def after_one_filter(filters):
    filtered_data_dic = {}
    for filter in filters:  
        filtered_data = data.copy()
        if "<=" in filter:
            column, value = filter.split("<=")
            filtered_data = filtered_data[filtered_data[column] <= float(value)]
            filtered_data_dic[filter] = filtered_data
        elif "=" in filter:
            column, value = filter.split("=")
            filtered_data = filtered_data[filtered_data[column] == value]
            filtered_data_dic[filter] = filtered_data
        else:
            column, value = filter.split(">")
            filtered_data = filtered_data[filtered_data[column] > float(value)]
            filtered_data_dic[filter] = filtered_data
    
    return filtered_data_dic

data_after_one_filter = after_one_filter(filters)

# after_one_filter(after overall is selected)
def after_two_filter(filters):
    topic_data_dic = {}
    for filter in filters:  
        if topic in filter:  
           topic_data_dic[filter] = data_after_one_filter[filter].copy()
    filtered_data_list = []
    for df in topic_data_dic.values():
        filtered_data_dic = {}
        for filter in filters:
            if topic not in filter: 
                if "<=" in filter:
                    column, value = filter.split("<=")
                    filtered_data = df[df[column] <= float(value)]
                    filtered_data_dic[filter] = filtered_data
                elif "=" in filter:
                    column, value = filter.split("=")
                    filtered_data = df[df[column] == value]
                    filtered_data_dic[filter] = filtered_data
                else:
                    column, value = filter.split(">")
                    filtered_data = df[df[column] > float(value)]
                    filtered_data_dic[filter] = filtered_data 
        filtered_data_list.append(filtered_data_dic)   
    return filtered_data_list
data_after_two_filter = after_two_filter(filters)  

# Prepare materials(parent and son) for plot bar chart (overall and with one-filter)
x_label =[]
y_label = []
y_label_2 =[]
topic_after_one_filter = [] 
def material_for_bar_chart(str):
    # parent(overall)
    for filter in filters:
        if str in filter:
            x_label.append(filter)
            y_label.append(data_after_one_filter[filter].shape[0]/data.shape[0]*100)
            topic_after_one_filter.append(data_after_one_filter[filter].shape[0])
    #  son(one-filter)  
    for filter in filters:
        temp = []
        if str in filter:
            continue
        else:
            for i in range(len(topic_after_one_filter)):
                for key in data_after_two_filter[i].keys():
                    if key == filter:
                        temp.append(data_after_two_filter[i][key].shape[0]/topic_after_one_filter[i]*100)
        y_label_2.append(temp)       
    return y_label, y_label_2

parent, children = material_for_bar_chart(topic)

# Calculate population_proportion of children (between Overall and one-filter bar charts)
row_sum = []
for i in range(len(data_after_two_filter)):
    temp = []
    for key in data_after_two_filter[i].keys():
        temp.append(data_after_two_filter[i][key].shape[0])
    row_sum.append(temp)
children_row_sum = list(sum(np.array(row_sum)[:,:]))

population_proportion = []
for i in range(len(children_row_sum)):
     population_proportion.append(children_row_sum[i]/data.shape[0]*100)

# Calculate Wasserstein distance(Earth mover’s distance) between Overall and one-filter bar charts
emd_dis = []
for i in range(len(children)):
    emd_dis.append(wasserstein_distance_nd(parent, children[i]))
utility = []
emd_dis_min = min(emd_dis)

# Use population_proportion as weight to multiply the distance to  get utility(U)
temp_dict = {}
for i in range(len(emd_dis)):
    if emd_dis[i] <= emd_dis_min/0.8:
        temp_dict[i] = population_proportion[i] * emd_dis[i]
utility.append(temp_dict)
utility_max = dict(sorted(temp_dict.items(), key = lambda x: x[1], reverse = True)[:3])
print(utility_max)

# Add bar height to bar chart
def addbarlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i],1), ha = 'center')

def plot_overall_bar_chart(x_label,y_label):
    color = ["#CDE8E5", "#77B0AA", "#135D66"]
    plt.bar(x_label,y_label,color=color)
    addbarlabels(x_label,y_label)
    plt.ylabel("Percentage")
    plt.yticks([0,25,50,75,100])
    plt.title("Overall")
    plt.box(False)
    plt.grid(axis = 'y', color="lightgray", linestyle="dashed", zorder=-10)
    plt.savefig("vispilot_img/img1.png",bbox_inches='tight',dpi=100)
    plt.close()
plot_overall_bar_chart(x_label,y_label)

def plot_one_filter_bar_chart(x_label,y_label_2):
    color = ["#CDE8E5", "#77B0AA", "#135D66"]
    for i in range(len(filters)):
        for key in utility_max.keys():
            if i == key:
                plt.bar(x_label,y_label_2[i],color=color)  
                addbarlabels(x_label,y_label_2[i])                                
                plt.ylabel("Percentage")
                plt.yticks([0,25,50,75,100])
                plt.title(filters[i])
                plt.box(False)
                plt.grid(axis = 'y', color="lightgray", linestyle="dashed", zorder=-10)
                plt.savefig(f"""vispilot_img/img{i+2}.png""",bbox_inches='tight',dpi=100)
                plt.close()
            else:
                continue
plot_one_filter_bar_chart(x_label,y_label_2)