FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./

RUN python3.11 -m pip install -r requirements.txt -t .

COPY . ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["data.lambda_handler"]