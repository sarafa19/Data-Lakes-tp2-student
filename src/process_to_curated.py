import argparse
import boto3
import pandas as pd
import io
import os
from transformers import AutoTokenizer
from botocore.exceptions import ClientError


def download_csv_from_s3(bucket_name, file_key, s3_client):
    print(f"Téléchargement de '{file_key}' depuis le bucket '{bucket_name}'...")
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def upload_file_to_s3(bucket_name, file_key, local_path, s3_client):
    print(f"Téléversement de '{file_key}' vers le bucket '{bucket_name}'...")
    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, bucket_name, file_key)


def tokenize_sequences_column(df, tokenizer, column="sequence", max_length=1024):
    print("Tokenisation des séquences...")
    tokens_list = df[column].apply(
        lambda seq: tokenizer(seq, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")["input_ids"][0]
    )

    token_columns = pd.DataFrame(tokens_list.tolist(), columns=[f"token_{i}" for i in range(max_length)])
    return pd.concat([df.drop(columns=[column]), token_columns], axis=1)

def ensure_bucket_exists(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created bucket '{bucket_name}'.")

def process_data(bucket_staging, bucket_curated, input_file, output_file, model_name):
    s3_client = boto3.client(
        's3',
        endpoint_url='http://localhost:4566',
        aws_access_key_id='root',
        aws_secret_access_key='root',
        region_name='us-east-1'
    )

    ensure_bucket_exists(s3_client, bucket_staging)
    ensure_bucket_exists(s3_client, bucket_curated)

    data = download_csv_from_s3(bucket_staging, input_file, s3_client)

    if "sequence" not in data.columns:
        raise KeyError("La colonne 'sequence' est absente du fichier d'entrée.")

    print(f"Chargement du tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Étape 3 : Tokenization
    processed_df = tokenize_sequences_column(data, tokenizer)

    temp_file_path = os.path.join("/tmp", output_file)
    processed_df.to_csv(temp_file_path, index=False)
    print(f"Données tokenisées sauvegardées localement dans '{temp_file_path}'.")

    # Étape 5 : Upload to curated bucket
    upload_file_to_s3(bucket_curated, output_file, temp_file_path, s3_client)
    print("Traitement terminé avec succès.")


if __name__ == "__main__":

    import sys
    sys.argv = [
        "process_to_curated.py",
        "--bucket_staging", "staging",
        "--bucket_curated", "curated",
        "--input_file", "preprocessed_train.csv", 
        "--output_file","tokenized_train.csv"
    ]
    parser = argparse.ArgumentParser(description="Préparation des données pour l'entraînement IA.")
    parser.add_argument("--bucket_staging", required=True, help="Nom du bucket staging")
    parser.add_argument("--bucket_curated", required=True, help="Nom du bucket curated")
    parser.add_argument("--input_file", required=True, help="Nom du fichier d'entrée")
    parser.add_argument("--output_file", required=True, help="Nom du fichier de sortie")
    parser.add_argument("--model_name", default="facebook/esm2_t6_8M_UR50D", help="Nom du tokenizer HuggingFace")

    args = parser.parse_args()
    process_data(
        bucket_staging=args.bucket_staging,
        bucket_curated=args.bucket_curated,
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name
    )
