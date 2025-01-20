import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lida import Manager, TextGenerationConfig, llm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from scipy.stats import wasserstein_distance_nd

def data_scope_generator(data_scope, data):
    # An example of data_scope[0] = "data['Origin']"
    index_start = data_scope[0].find("data['")
    measure_name =  data_scope[0][index_start + len("data['"):]
    index_end = measure_name.find("']")
    # measure_name = Origin
    measure_name = measure_name[:index_end]
    subspace_value = data[measure_name].unique()
    # The first data scope is the initial one
    new_subspace = []
    new_subspace.append(data_scope[0])
    # Tell subspace_value is a numeric value or a string value
    flag = False
    for i in range(0,len(subspace_value)):    
        if subspace_value.dtype in ["int64", "float64"]:
            flag = True
            new_subspace.append(f"data['{measure_name}'] = '" + str(subspace_value[i]) + "'" )
        else:
            new_subspace.append(f"data['{measure_name}'] = '{subspace_value[i]}'" )
    return new_subspace, flag

#  Preprocess data for calculate population_proportion Wasserstein distance(Earth mover’s distance)
def bar_data_generator(data_scope, flag, data):
    print(data)
    # An example of data_scope(Subspace, Breakdown, Measure):"data['Home_Type'] = 'Single Family'","Roof_Style", "data['Price'].mean()"
    filter_index_start = data_scope[0].find("data['")
    filter_value = ""
    if "=" not in data_scope[0]:
        filter_index = data_scope[0].find("']")
        filter_name =  data_scope[0][filter_index_start + len("data['"):filter_index]
        # e.g. data['Home_Type']
        total_row = len(data)
        # breakdown_name = Roof_Style
        if "data['" in data_scope[1]:
            index_start_1 = data_scope[1].find("data['")
            breakdown_name = data_scope[1][index_start_1 + len("data['"):]    
            index_end_1= breakdown_name.find("']")
            breakdown_name = breakdown_name[:index_end_1]
        else:
            breakdown_name = data_scope[1]
        # measure_name = Price
        index_start_2 = data_scope[2].find("data['")
        measure_name =  data_scope[2][index_start_2 + len("data['"):]
        index_end_2= measure_name.find("']")
        measure_name = measure_name[:index_end_2]
        if "mean" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'mean'}).reset_index()
        elif "median" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'median'}).reset_index()
        elif "sum" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'sum'}).reset_index()
        elif "count" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'count'}).reset_index()
        elif "max" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'max'}).reset_index()
        elif "min" in data_scope[2]:    
            data = data.groupby(breakdown_name).agg({measure_name: 'min'}).reset_index()
    else:
        equal_index = data_scope[0].find("=")
        filter_name =  data_scope[0][filter_index_start + len("data['"):equal_index-3]
        filter_value = data_scope[0][equal_index+2:]
        filter_value = filter_value.replace("'", "")
        # e.g. data['Home_Type'] = 'Single Family'
        if flag:
            filter_value = float(filter_value)
        data = data[data[filter_name] == filter_value]
        print(data)
        total_row = len(data)
        # breakdown_name = Roof_Style
        if "data['" in data_scope[1]:
            index_start_1 = data_scope[1].find("data['")
            breakdown_name = data_scope[1][index_start_1 + len("data['"):]
            index_end_1= breakdown_name.find("']")
            breakdown_name = breakdown_name[:index_end_1]
        else:
            breakdown_name = data_scope[1]
        # measure_name = Price
        index_start_2 = data_scope[2].find("data['")
        measure_name =  data_scope[2][index_start_2 + len("data['"):]
        index_end_2= measure_name.find("']")
        measure_name = measure_name[:index_end_2]
        print(data)
        if "mean" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'mean'}).reset_index()
        elif "median" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'median'}).reset_index()
        elif "sum" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'sum'}).reset_index()
        elif "count" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'count'}).reset_index()
        elif "max" in data_scope[2]:
            data = data.groupby(breakdown_name).agg({measure_name: 'max'}).reset_index()
        elif "min" in data_scope[2]:    
            data = data.groupby(breakdown_name).agg({measure_name: 'min'}).reset_index()

    return data, total_row, data[measure_name].values.tolist()


# Calculate Wasserstein distance(Earth mover’s distance) between the bar chart which has maximum population_proportion and others
def utility_calculator(arr_proportion, arr_for_EMD):
    utility = []
    emd_dis = []
    max_index = arr_proportion.index(max(arr_proportion))
    for i in range(len(arr_for_EMD)):
        if i != max_index:
            emd_dis.append(wasserstein_distance_nd(arr_for_EMD[max_index], arr_for_EMD[i]))

    emd_dis_min = min(emd_dis)
    emd_dis.insert(max_index, 0)
    # Use population_proportion as weight to multiply the distance to  get utility(U)
    temp_dict = {}
    for i in range(len(emd_dis)):
        if emd_dis[i] <= emd_dis_min/0.5:
            temp_dict[i] = arr_proportion[i] * emd_dis[i]
    utility.append(temp_dict)
    utility_max = dict(sorted(temp_dict.items(), key = lambda x: x[1], reverse = True)[:4])

    return list(utility_max.keys())
