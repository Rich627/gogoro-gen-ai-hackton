import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import boto3
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch,
)
from loguru import logger
from opensearchpy import OpenSearch
from promptflow.tools.aoai import tool

# logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

ENV_PATH = Path(__file__).parent / "env.local"
print(ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH)

AWS_PROFILE = os.environ.get("AWS_PROFILE")

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_EMBEDDINGS_DEPLOYMENT_NAME = os.environ.get("AZURE_EMBEDDINGS_DEPLOYMENT_NAME")

# OpenSearch
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")
OPENSEARCH_USERNAME = os.environ.get("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD")
RAG_THRESHOLD = float(os.environ.get("RAG_THRESHOLD", 0.5))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, default="What is the meaning of <3?")
    parser.add_argument("--index", type=str, default="movies")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument(
        "--bedrock-model-id",
        type=str,
        default="anthropic.claude-3-sonnet-20240229-v1:0",
    )
    parser.add_argument(
        "--bedrock-embedding-model-id", type=str, default="amazon.titan-embed-text-v1"
    )

    return parser.parse_known_args()


def get_model(model: str = "claude 3 sonnet") -> ChatBedrock:
    model = model.lower()
    if model == "claude 3 sonnet":
        llm = ChatBedrock(
            credentials_profile_name=AWS_PROFILE,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            streaming=True,
        )

    return llm


def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client


def create_langchain_vector_embedding_using_bedrock(
    bedrock_client, bedrock_embedding_model_id
):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client, model_id=bedrock_embedding_model_id
    )
    return bedrock_embeddings_client


def get_opensearch_client(cluster_url, username, password):

    client = OpenSearch(
        hosts=[cluster_url], http_auth=(username, password), verify_certs=True
    )
    return client


def create_opensearch_vector_search_client(
    index_name,
    bedrock_embeddings_client,
    opensearch_endpoint=OPENSEARCH_ENDPOINT,
    opensearch_username=OPENSEARCH_USERNAME,
    opensearch_password=OPENSEARCH_PASSWORD,
    _is_aoss=False,
):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=opensearch_endpoint,
        http_auth=(opensearch_username, opensearch_password),
        is_aoss=_is_aoss,
    )
    return docsearch


def create_index(opensearch_client, index_name):
    settings = {"settings": {"index": {"knn": True, "knn.space_type": "cosinesimil"}}}
    response = opensearch_client.indices.create(index=index_name, body=settings)
    return bool(response["acknowledged"])


def create_index_mapping(opensearch_client, index_name):
    response = opensearch_client.indices.put_mapping(
        index=index_name,
        body={
            "properties": {
                "vector_field": {"type": "knn_vector", "dimension": 1536},
                "text": {"type": "keyword"},
            }
        },
    )
    return bool(response["acknowledged"])


def delete_opensearch_index(opensearch_client, index_name):
    logger.info(f"Trying to delete index {index_name}")
    try:
        response = opensearch_client.indices.delete(index=index_name)
        logger.info(f"Index {index_name} deleted")
        return response["acknowledged"]
    except Exception as e:
        logger.info(f"Index {index_name} not found, nothing to delete")
        return True


@tool
def main(query: str, chat_history: List[Dict[str, Any]]):
    logger.info("Starting...")
    args, _ = parse_args()
    region = args.region
    index_name = args.index
    bedrock_model_id = args.bedrock_model_id
    bedrock_embedding_model_id = args.bedrock_embedding_model_id
    question = args.ask
    logger.info(f"Question provided: {query}")

    # Creating all clients for chain
    bedrock_client = get_bedrock_client(region)
    bedrock_llm = get_model()
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(
        bedrock_client, bedrock_embedding_model_id
    )
    opensearch_client = get_opensearch_client(
        OPENSEARCH_ENDPOINT, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD
    )
    opensearch_vector_search_client = create_opensearch_vector_search_client(
        index_name,
        bedrock_embeddings_client,
        OPENSEARCH_ENDPOINT,
        OPENSEARCH_USERNAME,
        OPENSEARCH_PASSWORD,
    )

    # LangChain prompt template
    prompt = ChatPromptTemplate.from_template(
        """If the context is not relevant, please answer the question by using your own knowledge about the topic. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content

    {context}

    Question: {input}
    Answer:"""
    )

    docs_chain = create_stuff_documents_chain(bedrock_llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=opensearch_vector_search_client.as_retriever(),
        combine_docs_chain=docs_chain,
    )

    logger.info(
        f"Invoking the chain with KNN similarity using OpenSearch, Bedrock FM {bedrock_model_id}, and Bedrock embeddings with {bedrock_embedding_model_id}"
    )
    response = retrieval_chain.invoke({"input": query})

    print("")
    logger.info(
        "These are the similar documents from OpenSearch based on the provided query:"
    )
    source_documents = response.get("context")
    for d in source_documents:
        print("")
        logger.info(f"Text: {d.page_content}")

    print("")
    logger.info(
        f"The answer from Bedrock {bedrock_model_id} is: {response.get('answer')}"
    )

    return response.get("answer")
