from langchain_openai import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage


def summary(openai_key):
    image_path1 = 'image_0.png'
    image_path2 = 'image_1.png'

    detail_parameter = 'high'

    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            HumanMessage(content='You are an excellent data scientist. Please write a paragraph as conclusions based on these charts. The conclusion must be verified by quoting relevant articles from your knowledge base (for example, providing the source of the article). Please do not use columnar formulas to write conclusions. Do not use special symbols other than English characters (such as *, `)'),
            HumanMessagePromptTemplate.from_template(
                [{'image_url': {'path': '{image_path1}', 'detail': '{detail_parameter}'}},
                {'image_url': {'path': '{image_path2}', 'detail': '{detail_parameter}'}},

                ]
            )
        ]
    )

    llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    gpt4_image_chain = chat_prompt_template | llm 
    response = gpt4_image_chain.invoke(input= {'image_path1':image_path1, 'image_path2':image_path2, 'detail_parameter': detail_parameter})
    return response.content

def description(image_path,openai_key):
    detail_parameter = 'high'

    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            HumanMessage(content='You are a good data scientist.Please describe this chart with no more than 100 words.'),
            HumanMessagePromptTemplate.from_template(
                [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}},
                ]
            )
        ]
    )

    llm = ChatOpenAI(model_name='gpt-4o-mini', api_key = openai_key)
    gpt4_image_chain = chat_prompt_template | llm 
    response = gpt4_image_chain.invoke(input= {'image_path':image_path,'detail_parameter': detail_parameter})
    return response.content
