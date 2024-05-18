import json
import boto3
import base64
from io import BytesIO
from PIL import Image

# Configuration
SERVICE_NAME = "bedrock-runtime"
REGION_NAME = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

def invoke_claude_3_multimodal(prompt: str, base64_image_data: str) -> str:
    """
    Invoke Anthropic Claude 3 Sonnet to perform multimodal inference using the provided input.

    :param prompt: The text prompt for Claude 3.
    :param base64_image_data: The base64-encoded image data to be included in the request.
    :return: The inferred response from the model.
    """
    
    client = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)

    # Create the request body
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}
                    },
                ],
            }
        ],
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
        )
        result = json.loads(response.get("body").read())
        return result['content'][0]['text'] if result.get("content") else ""
    except Exception as e:
        print(f"Error invoking the model: {type(e).__name__}: {e}")
        raise

def process_and_describe_image(image_path: str, prompt: str):
    # Open the image and convert it to base64
    with open(image_path, "rb") as image_file:
        buffered = BytesIO(image_file.read())
    base64_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Invoke the multimodal function
    description = invoke_claude_3_multimodal(prompt, base64_image_data)
    return description

# from langchain_core.prompts import ChatPromptTemplate
# chat_template = ChatPromptTemplate()

# chat_template.format_messages()

if __name__ == "__main__":
    
    image_path = "/Users/rich/Desktop/gen-ai-hackton/test.png"
    prompt ='''
    請用繁體描述這張圖, 並且取名為Description, 
    輸入格式限用這樣的格式: ![Image Descrption]({S3://gen-ai-hackton-image/}) 
    把Description換成你的描述, 後面的路徑不能換, 但是你可以自訂檔名'''
    
    description = process_and_describe_image(image_path, prompt)
    print(f"圖片描述: {description}")


