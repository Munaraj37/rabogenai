
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

# Environment variables
# Speech resource (required)
speech_region = os.environ.get('SPEECH_REGION') # e.g. westus2
speech_key = os.environ.get('SPEECH_KEY')
speech_private_endpoint = os.environ.get('SPEECH_PRIVATE_ENDPOINT') # e.g. https://my-speech-service.cognitiveservices.azure.com/ (optional)

# OpenAI resource (required for chat scenario)
azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') # e.g. https://my-aoai.openai.azure.com/
azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') # e.g. my-gpt-35-turbo-deployment
azure_openai_embed_deployment_name = os.environ.get('AZURE_OPENAI_EMBED_DEPLOYMENT_NAME') # e.g. my-gpt-35-turbo-deployment


# Cognitive search resource (optional, only required for 'on your data' scenario)
cognitive_search_endpoint = os.environ.get('COGNITIVE_SEARCH_ENDPOINT') # e.g. https://my-cognitive-search.search.windows.net/
cognitive_search_api_key = os.environ.get('COGNITIVE_SEARCH_API_KEY')
cognitive_search_index_name = os.environ.get('COGNITIVE_SEARCH_INDEX_NAME') # e.g. my-search-index

#service_endpoint = 'https://acsrabogenai.search.windows.net' #os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
#index_name = 'vrm-index' #os.environ["AZURE_SEARCH_INDEX_NAME"]
#key = 'eIvQly2BQv7MpjhhmCH0WYI7fBcqlIweVLecds5QLFAzSeBbtGBa' #os.environ["AZURE_SEARCH_API_KEY"]

#OpenAIEndpoint = 'https://aoairabogenai.openai.azure.com/'
#OpenAIKey = 'b1d16fbce07644668935410f4c54f74e'

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
    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2023-09-01-preview",
    )
    embedding = client.embeddings.create(input=[text], model=azure_openai_embed_deployment_name)
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
    index_client = SearchIndexClient(cognitive_search_endpoint, AzureKeyCredential(cognitive_search_api_key))
    index = get_hotel_index(cognitive_search_index_name)
    index_client.create_index(index)
    client = SearchClient(cognitive_search_endpoint, cognitive_search_index_name, AzureKeyCredential(cognitive_search_api_key))
    hotel_docs = get_document_info()
    print("Doc Loaded")
    client.upload_documents(documents=hotel_docs)
    print("Completed")
