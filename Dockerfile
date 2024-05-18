FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["data.lambda_handler"]