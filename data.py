import fitz
import boto3
import os
from io import BytesIO

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = 'gogoro-hackton-data'
    output_bucket_name = 'gogoro-hackton-data-source'

    # Extract the object key from the event
    key = event['Records'][0]['s3']['object']['key']
    filename = os.path.splitext(os.path.basename(key))[0]

    # Download the PDF file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    pdf_data = response['Body'].read()
    
    # Open the PDF file
    pdf_file = fitz.open(stream=pdf_data, filetype="pdf")

    # Create new PDF for modified content
    modified_pdf = fitz.open()

    # Set image minimum size threshold
    min_width, min_height = 20, 20

    # To store all pages' text content
    all_text = ""

    # Iterate through each page
    image_count = 1
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        modified_page = modified_pdf.new_page(width=page.rect.width, height=page.rect.height)

        # Get all image objects on the current page
        images = page.get_images(full=True)

        # Extract images and insert marks
        for img in images:
            xref = img[0]
            image_rects = page.get_image_rects(xref)

            for rect in image_rects:
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0

                # Only process images larger than the minimum size
                if width > min_width and height > min_height:
                    # Extract image data
                    image_data = pdf_file.extract_image(xref)
                    image_bytes = image_data["image"]
                    image_path = f"{filename}_image_{image_count}.png"

                    # Save image to S3
                    s3_client.put_object(Bucket=output_bucket_name, Key=image_path, Body=image_bytes)

                    # Insert text mark at the top-left corner of the image
                    mark_text = f"image {image_count}"
                    modified_page.insert_text(rect.top_left, mark_text, fontsize=12, color=(1, 0, 0))

                    # Add image mark to the plain text content
                    all_text += f"\n{mark_text}\n"

                    image_count += 1

        # Copy original page text content
        text = page.get_text()
        modified_page.insert_text((0, 0), text)
        
        # Accumulate text for all pages
        all_text += text + "\n"

    # Save modified PDF to a BytesIO object
    output_pdf_stream = BytesIO()
    modified_pdf.save(output_pdf_stream)
    output_pdf_stream.seek(0)

    # Upload modified PDF to S3 with a new name based on the original filename
    modified_pdf_key = f"{filename}.pdf"
    s3_client.put_object(Bucket=output_bucket_name, Key=modified_pdf_key, Body=output_pdf_stream.getvalue())
    modified_pdf.close()
    pdf_file.close()

    # Upload extracted text content to S3 with a new name based on the original filename
    extracted_text_key = f"{filename}.txt"
    s3_client.put_object(Bucket=output_bucket_name, Key=extracted_text_key, Body=all_text.encode('utf-8'))

    return {
        'statusCode': 200,
        'body': 'Images extracted and PDF modified successfully.'
    }
