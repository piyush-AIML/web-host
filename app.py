import torch
import torchvision.transforms as tf
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Model
# ================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128*8*8,256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,64),
            nn.ReLU(),

            nn.Linear(64,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

# Transform
transform = tf.Compose([
    tf.Resize((128,128)),
    tf.ToTensor(),
    tf.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

classes = ["Cat 🐱", "Dog 🐶"]

# ================================
# Helper
# ================================

def allowed_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))

# ================================
# Prediction
# ================================

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()

# ================================
# Routes
# ================================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and allowed_file(file.filename):

            # Unique filename (no overwrite)
            filename = str(uuid.uuid4()) + ".jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(path)

            prediction, confidence = predict_image(path)
            image_path = path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2) if confidence else None,
        image_path=image_path
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
