import os
import sys
import requests
from zipfile import ZipFile

def download_zip_file(url, output_dir):
    """Download a zip file from a URL to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        zip_path = os.path.join(output_dir, "heart_disease_data.zip")
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded zip file: {zip_path}")
        return zip_path
    else:
        raise Exception(f"Failed to download file: Status code {response.status_code}")


def extract_zip_file(zip_path, output_dir):
    """Extract all contents of a zip file and remove the zip."""
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted files to: {output_dir}")
    os.remove(zip_path)
    print("Removed the zip file.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 execute.py /path/to/extraction")
        sys.exit(1)

    EXTRACT_PATH = sys.argv[1]

    try:
        print("Starting Heart Disease Data Extraction Engine...")
        DATASET_URL = "https://storage.googleapis.com/kaggle-data-sets/888463/1508992/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250817%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250817T142305Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=18f9a059e4ade0d86c135c4e734b812a5085e3f1528321bb32c6a7e14c1be9f2831bf46b957f844ac412991cbea00d2556a3f55cb109995efa6a5d8e637a7f91d9fa15df115343e9e7c7007cdb723c2a42bfadcbf41c06cdc04eb0073fee28a65cc5bd45b2d219c8b5bb06745f9bca4e7250bf4350d7c61aa08d804da4f8260e8f9012d1803a55d48d17523ede46424295fe255ceb1c2ef328aa5cd239acea71618cbc78647fb82561df1d61d210e4a066d63d399c9d448b8601587d415e480331476de3ea62a761a364c707712ac563d0c294f09f5e81347a444313452e42bf94661d3e80405c263040947d13e176dc6da498a1617dbc23fd4da4c0f0667832"   
        zip_file = download_zip_file(DATASET_URL, EXTRACT_PATH)
        extract_zip_file(zip_file, EXTRACT_PATH)
        print("Extraction completed successfully! 🎉")
    except Exception as e:
        print(f"Error: {e}")


