import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedevastator/us-weather-history-12-months-of-record-setting-t")

print("Path to dataset files:", path)