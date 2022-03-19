import boto3
import os
from werkzeug.utils import secure_filename

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def upload_file_to_s3(file, acl="public-read", filename="processed_image.png"):
    # filename = secure_filename(file.filename)
    s3.upload_fileobj(
            file,
            os.getenv("AWS_BUCKET_NAME"),
            filename,
            ExtraArgs={
                "ACL": acl
                # "ContentType": file.content_type
            }
        )
        # after upload file to s3 bucket, return filename of the uploaded file
    return filename
