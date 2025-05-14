import gdown
import zipfile

def download_and_extract(file_id, output_zip='dataset.zip', extract_to='dataset'):

    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_zip, quiet=False)

    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Dane pobrane i rozpakowane do folderu: {extract_to}")
