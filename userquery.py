
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import openai

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

def get_embeddings(text: str):
    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2023-09-01-preview",
    )
    embedding = client.embeddings.create(input=[text], model=azure_openai_embed_deployment_name)
    return embedding.data[0].embedding


def get_hotel_index(query: str):
    
    search_client = SearchClient(cognitive_search_endpoint, cognitive_search_index_name, AzureKeyCredential(cognitive_search_api_key))   
    vector = VectorizedQuery(vector=get_embeddings(query),k_nearest_neighbors=2,fields="embedding")
    results = search_client.search(vector_queries=[vector],select=["content"])
    
    input_text=""
    for result in results:
        input_text = input_text + result['content'] + " "

    return input_text

if __name__ == "__main__":

    query = "What is the Qualifications for election as President"
    input_text = get_hotel_index(query)
    client = openai.AzureOpenAI(azure_endpoint=azure_openai_endpoint,api_key=azure_openai_api_key,api_version="2023-09-01-preview",)
    completion = client.completions.create(
        model = azure_openai_deployment_name,
        prompt = f"Answer Input:{input_text}. Question:{query}",
        #max_tokens=10,
        #top_p=1,
        #frequency_penalty=0,
        #presence_penalty=0
    )      

    print(completion.choices[0].text)
