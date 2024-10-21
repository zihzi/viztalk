import openai
import os
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

os.environ["OPENAI_API_KEY"] = 'sk-'

table = '''[HEAD]: Title|Worldwide Gross|Production Budget|Release Year|Content Rating|Running Time|Genre|Creative Type|Rotten Tomatoes Rating|IMDB Rating
---
[ROW] 1: From Dusk Till Dawn|25728961|20000000|1996|R|107|Horror|Fantasy|63|7.1
[ROW] 2: Broken Arrow|148345997|65000000|1996|R|108|Action|Contemporary Fiction|55|5.8
[ROW] 3: City Hall|20278055|40000000|1996|R|111|Drama|Contemporary Fiction|55|6.1
[ROW] 4: Happy Gilmore|38623460|10000000|1996|PG-13|92|Comedy|Contemporary Fiction|58|6.9
[ROW] 5: Fargo|51204567|7000000|1996|R|87|Thriller|Contemporary Fiction|94|8.3
[ROW] 6: The Craft|55669466|15000000|1996|R|100|Thriller|Fantasy|45|5.9
[ROW] 7: Twister|495900000|88000000|1996|PG-13|117|Action|Contemporary Fiction|57|6
[ROW] 8: Dragonheart|104364680|57000000|1996|PG-13|108|Adventure|Fantasy|50|6.2
[ROW] 9: The Phantom|17220599|45000000|1996|PG|100|Action|Super Hero|43|4.8
[ROW] 10: The Rock|336069511|75000000|1996|R|136|Action|Contemporary Fiction|66|7.2
'''

prompt_template_1 = '''
Here is the database table column header [HEAD] and user's query(the sentences between the <<<>>> symbols).
Based on the user's query and refering to [HEAD], the structure of the question should be organized as follows:["Data focus","Category","Color attribute","Chart type"].
"Data focus":The main piece of information user is interested in.Try to make it related to [HEAD].
"Category":The grouping or categorization attribute in [HEAD].
"Color attribute":The attribute used to color or differentiate segments within the chart.
"Chart type":If a chart type is specified, determine which types it is refering to the following list.If the chart type is not in the following list, chart type is "None".
A.bar
B.stacked bar
C.line
D.grouped line
E.scatter
F.grouped scatter
G.pie
Use the structure of the question to generate sentences which can enunciate user's query.

Example1:
[HEAD]: Model|MPG|Cylinders|Displacement|Horsepower|Weight|Acceleration|Year|Origin
user's query:vertical stacked bar of number of models by cylinders, colored by origin.
question structure:("data focus":"number of models", "category":"cylinders", "color attribute":"origin", "chart type":"stacked bar")
enhanced query:Draw a vertical stacked bar chart to show the number of car models grouped by the number of cylinders, with different colors representing the origin of the cars.

Example2:
[HEAD]:Title|Worldwide Gross|Production Budget|Release Year|Content Rating|Running Time|Genre|Creative Type|Rotten Tomatoes Rating|IMDB Rating
user's query:Which movie is the most popular?
question structure:("data focus":"average rating", "category":"movie","color attribute":"movie", "chart type":"None")
enhanced query:Draw a bar chart to show the average rating of each movie, combining Rotten Tomatoes Rating and IMDB Rating, and use different colors for each movie.

Example3:
[HEAD]:Lot Area|Lot Config|Home Type|Roof Style|Foundation Type|Basement Area|Heating Type|Central Air|Rooms|Fireplaces|Garage Type|Fence Type|Year|Price|Satisfaction
user's query:Show houses after 2009.
question structure:("data focus":"number of houses", "category":"houses", "color attribute":"year", "chart type":"None")
enhanced query:Draw a bar chart to show the number of houses built after 2009, with different colors representing each year.

Example4:
[HEAD]:product_id|product_type_code|product_name|product_price
user's query:Draw a scatter plot to demonstrate the price of clothes.
question structure:("data focus":"product_price", "category":"clothes", "color attribute":"clothes", "chart type":"scatter")
enhanced query:Create a scatter plot to show the prices of products where the product type is clothes. Use the product_price for the y-axis and product_id for the x-axis to display the data points.

user's query:<<<{question}>>>
question structure:("data focus":"", "category":"", "color attribute":"", "chart type":"")
enhanced query:
'''
prompt_1 = PromptTemplate.from_template(template=prompt_template_1)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
chain_1 = prompt_1 | llm | StrOutputParser()

prompt_template_2 = '''
The database table DF is shown as follows:\n{table}\n\nand the query based on the data above(the sentences between the <<<>>> symbols). Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

Example1:
The database table DF is shown as follows:
[HEAD]: Model|MPG|Cylinders|Displacement|Horsepower|Weight|Acceleration|Year|Origin
---
[ROW] 1: volkswagen 1131 deluxe sedan|26|4|97|46|1835|20.5|1970|Europe
[ROW] 2: volkswagen super beetle|26|4|97|46|1950|21|1973|Europe
[ROW] 3: volkswagen rabbit custom diesel|43.1|4|90|48|1985|21.5|1978|Europe
[ROW] 4: vw rabbit c (diesel)|44.3|4|90|48|2085|21.7|1980|Europe
[ROW] 5: vw dasher (diesel)|43.4|4|90|48|2335|23.7|1980|Europe
[ROW] 6: fiat 128|29|4|68|49|1867|19.5|1973|Europe
[ROW] 7: toyota corona|31|4|76|52|1649|16.5|1974|Japan
[ROW] 8: chevrolet chevette|29|4|85|52|2035|22.2|1976|US
[ROW] 9: mazda glc deluxe|32.8|4|78|52|1985|19.4|1978|Japan
[ROW] 10: vw pickup|44|4|97|52|2130|24.6|1982|Europe

Provide the table which can transform to visual charts to meet the query: "Draw a vertical stacked bar chart to show the number of car models grouped by the number of cylinders, with different colors representing the origin of the cars.".Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

SQL: ```SELECT  Model,Cylinders,Origin FROM DF;```.

The database table DF is shown as follows:
[HEAD]: Model|Cylinders|Origin
---
[ROW] 1: volkswagen 1131 deluxe sedan|4|Europe
[ROW] 2: volkswagen super beetle|4|Europe
[ROW] 3: volkswagen rabbit custom diesel|4|Europe
[ROW] 4: vw rabbit c (diesel)|4|Europe
[ROW] 5: vw dasher (diesel)|4|Europe
[ROW] 6: fiat 128|4|Europe
[ROW] 7: toyota corona|4|Japan
[ROW] 8: chevrolet chevette|4|US
[ROW] 9: mazda glc deluxe|4|Japan
[ROW] 10: vw pickup|4|Europe

Provide the table which can transform to visual charts to meet the query: "Draw a vertical stacked bar chart to show the number of car models grouped by the number of cylinders, with different colors representing the origin of the cars.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

SQL: ```SELECT Cylinders, Origin, COUNT(Model) as NumberOfModels FROM DF GROUP BY Cylinders, Origin;```.

The database table DF is shown as follows:
[HEAD]: Cylinders|Origin|NumberOfModels
---
[ROW] 1: 4|Europe|7
[ROW] 2: 4|Japan|2
[ROW] 3: 4|US|1

Provide the table which can transform to visual charts to meet the query: "Draw a vertical stacked bar chart to show the number of car models grouped by the number of cylinders, with different colors representing the origin of the cars.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

final table:
 ```
[HEAD]: Cylinders|Origin|NumberOfModels
---
[ROW] 1: 4|Europe|7
[ROW] 2: 4|Japan|2
[ROW] 3: 4|US|1
```.


Example2:
The database table DF is shown as follows:
[HEAD]: Title|Worldwide Gross|Production Budget|Release Year|Content Rating|Running Time|Genre|Creative Type|Rotten Tomatoes Rating|IMDB Rating
---
[ROW] 1: From Dusk Till Dawn|25728961|20000000|1996|R|107|Horror|Fantasy|63|7.1
[ROW] 2: Broken Arrow|148345997|65000000|1996|R|108|Action|Contemporary Fiction|55|5.8
[ROW] 3: City Hall|20278055|40000000|1996|R|111|Drama|Contemporary Fiction|55|6.1
[ROW] 4: Happy Gilmore|38623460|10000000|1996|PG-13|92|Comedy|Contemporary Fiction|58|6.9
[ROW] 5: Fargo|51204567|7000000|1996|R|87|Thriller|Contemporary Fiction|94|8.3
[ROW] 6: The Craft|55669466|15000000|1996|R|100|Thriller|Fantasy|45|5.9
[ROW] 7: Twister|495900000|88000000|1996|PG-13|117|Action|Contemporary Fiction|57|6
[ROW] 8: Dragonheart|104364680|57000000|1996|PG-13|108|Adventure|Fantasy|50|6.2
[ROW] 9: The Phantom|17220599|45000000|1996|PG|100|Action|Super Hero|43|4.8
[ROW] 10: The Rock|336069511|75000000|1996|R|136|Action|Contemporary Fiction|66|7.2

Provide the table which can transform to visual charts to meet the query: "Draw a bar chart to show the average rating of each movie, combining Rotten Tomatoes Rating and IMDB Rating, and use different colors for each movie.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

Python: ```
DF['Average Rating'] = DF[['Rotten Tomatoes Rating', 'IMDB Rating']].mean(axis=1)
```.

The database table DF is shown as follows:
[HEAD]: Title|Worldwide Gross|Production Budget|Release Year|Content Rating|Running Time|Genre|Creative Type|Rotten Tomatoes Rating|IMDB Rating|Average Rating
---
[ROW] 1: From Dusk Till Dawn|25728961|20000000|1996|R|107|Horror|Fantasy|63|7.1|35.05
[ROW] 2: Broken Arrow|148345997|65000000|1996|R|108|Action|Contemporary Fiction|55|5.8|30.4
[ROW] 3: City Hall|20278055|40000000|1996|R|111|Drama|Contemporary Fiction|55|6.1|30.55
[ROW] 4: Happy Gilmore|38623460|10000000|1996|PG-13|92|Comedy|Contemporary Fiction|58|6.9|32.45
[ROW] 5: Fargo|51204567|7000000|1996|R|87|Thriller|Contemporary Fiction|94|8.3|51.15
[ROW] 6: The Craft|55669466|15000000|1996|R|100|Thriller|Fantasy|45|5.9|25.45
[ROW] 7: Twister|495900000|88000000|1996|PG-13|117|Action|Contemporary Fiction|57|6|31.75
[ROW] 8: Dragonheart|104364680|57000000|1996|PG-13|108|Adventure|Fantasy|50|6.2|28.1
[ROW] 9: The Phantom|17220599|45000000|1996|PG|100|Action|Super Hero|43|4.8|23.9
[ROW] 10: The Rock|336069511|75000000|1996|R|136|Action|Contemporary Fiction|66|7.2|36.6

Provide the table which can transform to visual charts to meet the query: "Draw a bar chart to show the average rating of each movie, combining Rotten Tomatoes Rating and IMDB Rating, and use different colors for each movie.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

SQL: ```SELECT  Title,Average Rating FROM DF;```.

The database table DF is shown as follows:
[HEAD]: Title|Average Rating
---
[ROW] 1: From Dusk Till Dawn|35.05
[ROW] 2: Broken Arrow|30.4
[ROW] 3: City Hall|30.55
[ROW] 4: Happy Gilmore|32.45
[ROW] 5: Fargo|51.15
[ROW] 6: The Craft|25.45
[ROW] 7: Twister|31.75
[ROW] 8: Dragonheart|28.1
[ROW] 9: The Phantom|23.9
[ROW] 10: The Rock|36.6

Provide the table which can transform to visual charts to meet the query: "Draw a bar chart to show the average rating of each movie, combining Rotten Tomatoes Rating and IMDB Rating, and use different colors for each movie.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

SQL: ```SELECT * FROM DF ORDER BY Average Rating DESC;```.

The database table DF is shown as follows:
[HEAD]: Title||Average Rating
---
[ROW] 1: Fargo|51.15
[ROW] 2: The Rock|36.6
[ROW] 3: From Dusk Till Dawn|35.05
[ROW] 4: Happy Gilmore|32.45
[ROW] 5: Twister|31.75
[ROW] 6: City Hall|30.55
[ROW] 7: Broken Arrow|30.4
[ROW] 8: Dragonheart|28.1
[ROW] 9: The Craft|25.45
[ROW] 10: The Phantom|23.9

Provide the table which can transform to visual charts to meet the query: "Draw a bar chart to show the average rating of each movie, combining Rotten Tomatoes Rating and IMDB Rating, and use different colors for each movie.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

final table:
 ```
[HEAD]: Title|Average Rating
---
[ROW] 1: Fargo|51.15
[ROW] 2: The Rock|36.6
[ROW] 3: From Dusk Till Dawn|35.05
[ROW] 4: Happy Gilmore|32.45
[ROW] 5: Twister|31.75
[ROW] 6: City Hall|30.55
[ROW] 7: Broken Arrow|30.4
[ROW] 8: Dragonheart|28.1
[ROW] 9: The Craft|25.45
[ROW] 10: The Phantom|23.9
```.

Example3:
The database table DF is shown as follows:
[HEAD]: product_id|product_type_code|product_name|product_price
---
[ROW] 1: 1|Clothes|red jeans|734.73
[ROW] 2: 2|Clothes|yellow jeans|687.23
[ROW] 3: 3|Clothes|black jeans|695.16
[ROW] 4: 4|Clothes|blue jeans|939.57
[ROW] 5: 5|Clothes|red jeans|534.52
[ROW] 6: 6|Clothes|red topping|408.82
[ROW] 7: 7|Clothes|black topping|916.53
[ROW] 8: 8|Clothes|yellow topping|918.41
[ROW] 9: 9|Clothes|blue topping|604.86
[ROW] 10: 10|Hardware|monitor|813.76

Provide the table which can transform to visual charts to meet the query: "Create a scatter plot to show the prices of products where the product type is clothes. Use the product_price for the y-axis and product_id for the x-axis to display the data points.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

SQL: ```
SELECT product_type_code, product_price FROM DF WHERE product_type_code = 'Clothes';
```.

The database table DF is shown as follows:
[HEAD]: product_type_code|product_price
---
[ROW] 1: Clothes|734.73
[ROW] 2: Clothes|687.23
[ROW] 3: Clothes|695.16
[ROW] 4: Clothes|939.57
[ROW] 5: Clothes|534.52
[ROW] 6: Clothes|408.82
[ROW] 7: Clothes|916.53
[ROW] 8: Clothes|918.41
[ROW] 9: Clothes|604.86

Provide the table which can transform to visual charts to meet the query: "Create a scatter plot to show the prices of products where the product type is clothes. Use the product_price for the y-axis and product_id for the x-axis to display the data points.". Execute SQL or Python code step-by-step and finally provide the table which can transform to visual charts to meet the query. Choose from generating a SQL, Python code, or directly providing the table.Jsut to provide the final table and DO NOT make any comments.

final table:
 ```
[HEAD]: product_type_code|product_price
---
[ROW] 1: Clothes|734.73
[ROW] 2: Clothes|687.23
[ROW] 3: Clothes|695.16
[ROW] 4: Clothes|939.57
[ROW] 5: Clothes|534.52
[ROW] 6: Clothes|408.82
[ROW] 7: Clothes|916.53
[ROW] 8: Clothes|918.41
[ROW] 9: Clothes|604.86
```.
query:<<<{enhance_query}>>>
final table:
'''

prompt_2 = PromptTemplate.from_template(template=prompt_template_2)
chain_2 =  {'enhance_query': RunnablePassthrough(), 'table':RunnablePassthrough()} | prompt_2 | llm | StrOutputParser()

chat_prompt = ChatPromptTemplate.from_template("Generate Python Code Script to visualize the data.")
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chat_chain = chat_prompt | chat_llm | StrOutputParser()

def format_question(question,dataset):
    question_tollm = question
    dataset_tollm = dataset
    return  question_tollm,dataset_tollm

question = format_question()  # Define the "question" variable


final_chain = ({"Chain1":chain_1}
               | RunnablePassthrough.assign(chain2=chain_2)
               | RunnablePassthrough.assign(chatchain=chat_chain))
answer = final_chain.invoke({"question":question})
print(answer)
