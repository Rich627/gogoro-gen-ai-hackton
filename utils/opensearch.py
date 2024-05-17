import os
import sys

import boto3
from dotenv import load_dotenv
from loguru import logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

load_dotenv()

# logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

# OpenSearch
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")


def get_opensearch_cluster_client(name, password, region):
    opensearch_endpoint = OPENSEARCH_ENDPOINT
    opensearch_client = OpenSearch(
        hosts=[opensearch_endpoint],
        http_auth=(name, password),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )
    return opensearch_client


def put_bulk_in_opensearch(list, client):
    logger.info(f"Putting {len(list)} documents in OpenSearch")
    success, failed = bulk(client, list)
    return success, failed


def check_opensearch_index(opensearch_client, index_name):
    return opensearch_client.indices.exists(index=index_name)


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
