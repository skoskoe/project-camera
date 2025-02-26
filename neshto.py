import requests

url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/yolov8n-face.pt"
filename = "yolov8n-face.pt"

response = requests.get(url, stream=True)
with open(filename, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

print("Download complete: yolov8n-face.pt")

from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")  # This should automatically download the correct model
