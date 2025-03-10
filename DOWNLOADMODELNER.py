import gdown

file_id = "1ZIgEkkF7nnMpakG0Z13WaX9Gb-__SJCz"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

output = "model.safetensors"
print("Downloading model...")
gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=True)
print("Download completed!")