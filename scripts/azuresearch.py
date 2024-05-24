
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import Vector

service_endpoint = 'https://acsrabogenai.search.windows.net' #os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = 'vrm-index' #os.environ["AZURE_SEARCH_INDEX_NAME"]
key = 'eIvQly2BQv7MpjhhmCH0WYI7fBcqlIweVLecds5QLFAzSeBbtGBa' #os.environ["AZURE_SEARCH_API_KEY"]

OpenAIEndpoint = 'https://aoairabogenai.openai.azure.com/'
OpenAIKey = 'b1d16fbce07644668935410f4c54f74e'

def get_document_info():

    import json
    import uuid
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = PyPDFLoader(r"D:\Python\input\Constitution.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    docs = []
    for doc in documents:
        
        docs.append({"documentID":str(uuid.uuid4()),"content":doc.page_content,"embedding":get_embeddings(doc.page_content)})
        
    json_data=json.dumps(docs)
    
    with open(r"D:\Python\output\HandbookContent.json","w") as f:
        f.write(json_data)

    with open(r"D:\Python\output\HandbookContent.json","r") as f:
        document = json.load(f)

    return document


def get_embeddings(text: str):
    # There are a few ways to get embeddings. This is just one example.
    import openai

    open_ai_endpoint = OpenAIEndpoint #os.getenv("OpenAIEndpoint")
    open_ai_key = OpenAIKey #os.getenv("OpenAIKey")

    client = openai.AzureOpenAI(
        azure_endpoint=open_ai_endpoint,
        api_key=open_ai_key,
        api_version="2023-09-01-preview",
    )
    embedding = client.embeddings.create(input=[text], model="embedding-rabogenai")
    return embedding.data[0].embedding


def get_hotel_index(name: str):
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

    fields = [
        SimpleField(name="documentID", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        )
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


if __name__ == "__main__":
    print("Started")
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(service_endpoint, credential)
    index = get_hotel_index(index_name)
    index_client.create_index(index)
    print("Index Config")
    client = SearchClient(service_endpoint, index_name, credential)
    hotel_docs = get_document_info()
    print("Doc Loaded")
    #
    print(hotel_docs)
    client.upload_documents(documents=hotel_docs)
    print("Completed")

query = "What is the preamble of Indian Constitution"
vector = Vector()