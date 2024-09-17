import os
import weaviate
import streamlit as st
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
import weaviate.classes as weaviate_classes

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_APIKEY"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_wcs(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers={
        "X-Cohere-Api-Key": cohere_api_key
    }
)

