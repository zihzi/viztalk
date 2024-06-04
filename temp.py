from langchain_community.document_loaders import CSVLoader 

loader = CSVLoader("department_store.csv")

data = loader.load()
print(data)