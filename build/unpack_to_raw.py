import os
import pandas as pd
import boto3
from botocore.exceptions import ClientError


def unpack_data(input_dir, bucket_name, output_file_name):
    """
    Unpacks and combines multiple CSV files from train, test, and dev subfolders into a single CSV file,
    then uploads the combined file to the specified S3 bucket.

    Parameters:
    input_dir (str): Path to the directory containing the train, test, and dev subfolders.
    bucket_name (str): Name of the S3 bucket to upload the combined file to.
    output_file_name (str): Name of the combined CSV file to be uploaded to S3.
    """
    s3 = boto3.client('s3', 
                    endpoint_url='http://localhost:4566',
                    aws_access_key_id='root',
                    aws_secret_access_key='root',
                    region_name='us-east-1',
                    )

    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created bucket '{bucket_name}'.")

    data_frames = []

    # Iterate through train, test, and dev subfolders
    for subfolder in ['train', 'test', 'dev']:
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                print(f"Reading {file_path}")
                data = pd.read_csv(
                    file_path,
                    names=['sequence', 'family_accession', 'sequence_name', 'aligned_sequence', 'family_id']
                )
                data_frames.append(data)
        else:
            print(f"Subfolder {subfolder_path} does not exist or is not a directory.")

    # Combine all data frames into a single data frame
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=True)
        print("All files combined successfully.")

        # Save the combined data to a CSV file
        tmp_dir = "./Data-Lakes-tp2-student/data/raw"
        os.makedirs(tmp_dir, exist_ok=True)
        combined_csv_path = os.path.join(tmp_dir, output_file_name)
        combined_data.to_csv(combined_csv_path, index=False)
        print(f"Combined file saved locally at {combined_csv_path}.")

        # Upload the combined file to the S3 bucket
        s3.upload_file(combined_csv_path, bucket_name, output_file_name)
        print(f"Uploaded combined file to bucket '{bucket_name}' with name '{output_file_name}'.")
    else:
        print("No valid files found to process.")

    #  Delete the local file after upload
    os.remove(combined_csv_path)
    print(f"Deleted local file: {combined_csv_path}")


if __name__ == "__main__":
    import argparse
    print('Script started')

    import sys
    sys.argv = [
        "unpack_data.py",
        "--input_dir", "./Data-Lakes-tp2-student/data/raw",
        "--bucket_name", "raw",
        "--output_file_name", "combined_raw.csv", 
    ]

    parser = argparse.ArgumentParser(description="Unpack, combine, and upload protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--bucket_name", type=str, required=True, help="Name of the S3 bucket")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the output file for S3")
    args = parser.parse_args()


    unpack_data(args.input_dir, args.bucket_name, args.output_file_name)
