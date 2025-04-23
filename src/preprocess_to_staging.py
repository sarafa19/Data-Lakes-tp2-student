import io
import pandas as pd
import boto3
import numpy as np
import tqdm
import joblib
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from numba import njit
from botocore.exceptions import ClientError


#@njit
def split_data_func(family_accession, class_encoded, test_ratio=0.33, dev_ratio=0.33):
    """
    Splits data into train, dev, and test indices based on unique classes.

    Parameters:
    family_accession (np.ndarray): Array of class labels (string or int).
    class_encoded (np.ndarray): Array of encoded class labels (integers).
    test_ratio (float): Ratio of data to allocate to the test set.
    dev_ratio (float): Ratio of remaining data to allocate to the dev set.

    Returns:
    (np.ndarray, np.ndarray, np.ndarray): Indices for train, dev, and test sets.
    """
    max_samples = len(family_accession)
    train_indices = np.empty(max_samples, dtype=np.int64)
    dev_indices = np.empty(max_samples, dtype=np.int64)
    test_indices = np.empty(max_samples, dtype=np.int64)

    train_count = 0
    dev_count = 0
    test_count = 0

    unique_classes = np.unique(family_accession)


    for i in range(len(unique_classes)):
        cls = unique_classes[i]
        print(f"Processing class {i + 1}/{len(unique_classes)} (label: {cls})")
        class_indices = np.where(family_accession == cls)[0]
        count = len(class_indices)

        if count == 1:
            test_indices[test_count:test_count+1] = class_indices
            test_count += 1
        elif count == 2:
            dev_indices[dev_count:dev_count+1] = class_indices[0:1]
            test_indices[test_count:test_count+1] = class_indices[1:2]
            dev_count += 1
            test_count += 1
        elif count == 3:
            train_indices[train_count:train_count+1] = class_indices[0:1]
            dev_indices[dev_count:dev_count+1] = class_indices[1:2]
            test_indices[test_count:test_count+1] = class_indices[2:3]
            train_count += 1
            dev_count += 1
            test_count += 1
        else:
            shuffled = np.random.permutation(class_indices)
            num_test = int(count * test_ratio)
            num_dev = int((count - num_test) * dev_ratio)
            num_train = count - num_test - num_dev

            test_indices[test_count:test_count+num_test] = shuffled[0:num_test]
            dev_indices[dev_count:dev_count+num_dev] = shuffled[num_test:num_test+num_dev]
            train_indices[train_count:train_count+num_train] = shuffled[num_test+num_dev:]

            test_count += num_test
            dev_count += num_dev
            train_count += num_train

    return (train_indices[:train_count], dev_indices[:dev_count], test_indices[:test_count])


def ensure_bucket_exists(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Created bucket '{bucket_name}'.")


def preprocess_to_staging(bucket_raw, bucket_staging, input_file, output_prefix):
    """
    Preprocesses data from the raw bucket and uploads preprocessed data splits to the staging bucket.

    Steps:
    1. Downloads the raw data file from the raw bucket.
    2. Cleans the data (handles missing values).
    3. Encodes the 'family_accession' column into numeric labels.
    4. Splits the data into train, dev, and test sets.
    5. Uploads the preprocessed data splits (train, dev, test) to the staging bucket.
    6. Saves metadata like label encodings and class weights to the staging bucket.

    Parameters:
    bucket_raw (str): Name of the raw S3 bucket.
    bucket_staging (str): Name of the staging S3 bucket.
    input_file (str): Name of the input file in the raw bucket.
    output_prefix (str): Prefix for the preprocessed output files in the staging bucket.
    """

    s3 = boto3.client('s3', 
                    endpoint_url='http://localhost:4566',
                    aws_access_key_id='root',
                    aws_secret_access_key='root',
                    region_name='us-east-1',
                    )
    
    ensure_bucket_exists(s3, bucket_raw)
    ensure_bucket_exists(s3, bucket_staging)

    # Step 1: Download raw data
    response = s3.get_object(Bucket=bucket_raw, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))

    # Step 2: Handle missing values
    print("Cleaning data by removing missing values...")
    data = data.dropna()

    # Step 3: Encode categorical labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Save the label encoder mapping
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    label_mapping_csv = pd.DataFrame(list(label_mapping.items()), columns=['family_accession', 'class_encoded'])
    csv_buffer = io.StringIO()
    label_mapping_csv.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_staging,
        Key=f"{output_prefix}_label_mapping.csv",
        Body=csv_buffer.getvalue()
    )
    print("Label mapping saved to staging bucket.")

    # Step 4: Split data into train, dev, and test sets
    print("Splitting data into train, dev, and test sets...")
    family_accession = data['family_accession'].astype('category').cat.codes.values
    class_encoded = data['class_encoded'].values

    family_accession = np.array(family_accession)
    class_encoded = np.array(class_encoded)

    train_indices, dev_indices, test_indices = split_data_func(family_accession, class_encoded)

    # Create DataFrames for each split
    train_data = data.iloc[train_indices]
    dev_data = data.iloc[dev_indices]
    test_data = data.iloc[test_indices]

    # Drop unnecessary columns
    train_data = train_data.drop(columns=["family_id", "sequence_name", "family_accession"])
    dev_data = dev_data.drop(columns=["family_id", "sequence_name", "family_accession"])
    test_data = test_data.drop(columns=["family_id", "sequence_name", "family_accession"])

    # Step 5: Upload preprocessed splits to staging bucket
    for split_name, split_data in zip(['train', 'dev', 'test'], [train_data, dev_data, test_data]):
        csv_buffer = io.StringIO()
        split_data.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket_staging,
            Key=f"{output_prefix}_{split_name}.csv",
            Body=csv_buffer.getvalue()
        )
        print(f"{split_name.capitalize()} split uploaded to staging bucket.")

    # Step 6: Calculate and save class weights
    print("Calculating class weights...")
    class_counts = train_data['class_encoded'].value_counts()
    class_weights = 1. / class_counts
    class_weights /= class_weights.sum()

    # Scale weights
    min_weight = class_weights.max()
    weight_scaling_factor = 1 / min_weight
    class_weights *= weight_scaling_factor

    # Save class weights
    class_weights_dict = OrderedDict(sorted(class_weights.items()))
    class_weights_csv = pd.DataFrame(list(class_weights_dict.items()), columns=['class', 'weight'])
    csv_buffer = io.StringIO()
    class_weights_csv.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_staging,
        Key=f"{output_prefix}_class_weights.csv",
        Body=csv_buffer.getvalue()
    )
    print("Class weights saved to staging bucket.")


if __name__ == "__main__":
    import argparse

    import sys
    sys.argv = [
        "preprocess_to_staging.py",
        "--bucket_raw", "raw",
        "--bucket_staging", "staging",
        "--input_file", "combined_raw.csv",
        "--output_prefix", "preprocessed", 
    ]
    parser = argparse.ArgumentParser(description="Preprocess data from raw to staging bucket")
    parser.add_argument("--bucket_raw", type=str, required=True, help="Name of the raw S3 bucket")
    parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in raw bucket")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files in staging bucket")
    args = parser.parse_args()

    preprocess_to_staging(args.bucket_raw, args.bucket_staging, args.input_file, args.output_prefix)
