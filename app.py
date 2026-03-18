import torch
import torchvision.transforms as tf
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
import os
import uuid

# ================================
# App Config
# ================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 🔥 Force CPU + limit threads (important for Render)
device = torch.device("cpu")
torch.set_num_threads(1)

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

# ================================
# Load Model (only once)
# ================================

model = CNN().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location="cpu"))
model.eval()

# ================================
# Transform
# ================================

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
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # 🔥 reduce memory usage early
            img.thumbnail((128, 128))

            image = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        return classes[pred.item()], conf.item()

    except Exception as e:
        return "Error", 0.0

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

            filename = f"{uuid.uuid4()}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(path)

                prediction, confidence = predict_image(path)
                image_path = path

            finally:
                # 🧹 cleanup to prevent storage/memory issues
                if os.path.exists(path):
                    os.remove(path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2) if confidence else None,
        image_path=image_path
    )

# ================================
# Entry (for local only)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
