from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the wildfire model
class WildfireModel(nn.Module):
    def __init__(self):
        super(WildfireModel, self).__init__()
        self.cnn = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.cnn(x)

# Load the trained model
model = WildfireModel().to(device)
model.load_state_dict(torch.load("wildfire_model_optimized.pth", map_location=device))
model.eval()

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction function
def predict_fire(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).squeeze()
        prediction = torch.sigmoid(output).item()
    return "🔥 Wildfire Detected!" if prediction > 0.5 else "✅ No Wildfire"

# Index route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            prediction = predict_fire(path)
            image_url = f"/{UPLOAD_FOLDER}/{file.filename}"  # For browser access

    return render_template("index.html", prediction=prediction, image_url=image_url)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
