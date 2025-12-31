#!/usr/bin/env python3
"""
MNIST Digit Recognition Web Demo
================================
A minimal FastAPI web application for testing the trained CNN model.

Author: Bahman Farhadian
Usage:
    pip install fastapi uvicorn pillow
    python app.py

Then open http://localhost:8000 in your browser.
"""

import base64
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("Please install required packages:")
    print("  pip install fastapi uvicorn pillow")
    exit(1)


# =============================================================================
# Model Definition (same as src/model.py)
# =============================================================================
class DigitRecognizer(nn.Module):
    """CNN for digit classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 320)
        self.fc2 = nn.Linear(320, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(title="MNIST Digit Recognition", version="1.0.0")

# Global state
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = {"correct": 0, "wrong": 0}


def load_model():
    """Load the trained model."""
    global model
    model = DigitRecognizer()
    
    model_path = Path("outputs/model.pt")
    if not model_path.exists():
        model_path = Path("model.pt")
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded from {model_path}")
    else:
        print("⚠ Warning: No model.pt found. Run training first!")
    
    model.to(device)
    model.eval()


def preprocess_image(image_data: str) -> torch.Tensor:
    """Convert base64 canvas image to model input tensor."""
    # Decode base64
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    
    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Create white background and paste image
    background = Image.new("RGB", img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[3])
    
    # Convert to grayscale
    img = background.convert("L")
    
    # Invert (canvas is white-on-black, MNIST is white digit on black)
    img = ImageOps.invert(img)
    
    # Find bounding box and crop
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        # Add padding
        width, height = img.size
        max_dim = max(width, height)
        # Create square image with padding
        new_img = Image.new("L", (max_dim + 40, max_dim + 40), 0)
        paste_x = (max_dim + 40 - width) // 2
        paste_y = (max_dim + 40 - height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize (MNIST normalization)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor - 0.1307) / 0.3081
    
    return img_tensor.to(device)


# =============================================================================
# HTML Template
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e7;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            color: #71717a;
            font-size: 0.9rem;
            margin-bottom: 30px;
        }
        
        .container {
            display: flex;
            gap: 40px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: flex-start;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
        }
        
        .draw-section {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .canvas-label {
            font-size: 0.85rem;
            color: #a1a1aa;
            margin-bottom: 12px;
        }
        
        #canvas {
            background: #000;
            border-radius: 12px;
            cursor: crosshair;
            touch-action: none;
        }
        
        .buttons {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }
        
        button {
            padding: 12px 28px;
            font-size: 0.95rem;
            font-weight: 500;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .btn-clear {
            background: #3f3f46;
            color: #e4e4e7;
        }
        
        .btn-clear:hover {
            background: #52525b;
        }
        
        .btn-check {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
        }
        
        .btn-check:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        }
        
        .result-section {
            min-width: 280px;
        }
        
        .result-section h2 {
            font-size: 1rem;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 16px;
        }
        
        .result-box {
            background: #000;
            border: 1px solid #3f3f46;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
        }
        
        .predicted-digit {
            font-size: 4rem;
            font-weight: 700;
            color: #60a5fa;
            line-height: 1;
        }
        
        .confidence {
            font-size: 0.9rem;
            color: #71717a;
            margin-top: 8px;
        }
        
        .feedback-section {
            margin-top: 16px;
        }
        
        .feedback-label {
            font-size: 0.85rem;
            color: #a1a1aa;
            margin-bottom: 8px;
        }
        
        .feedback-buttons {
            display: flex;
            gap: 8px;
        }
        
        .btn-correct, .btn-wrong {
            flex: 1;
            padding: 10px;
            font-size: 0.9rem;
        }
        
        .btn-correct {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .btn-correct:hover {
            background: rgba(34, 197, 94, 0.3);
        }
        
        .btn-wrong {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .btn-wrong:hover {
            background: rgba(239, 68, 68, 0.3);
        }
        
        .stats-section {
            margin-top: 24px;
        }
        
        .stats-section h2 {
            font-size: 1rem;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 12px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }
        
        .stat-item {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .stat-value.correct { color: #4ade80; }
        .stat-value.wrong { color: #f87171; }
        .stat-value.accuracy { color: #60a5fa; }
        
        .stat-label {
            font-size: 0.75rem;
            color: #71717a;
            margin-top: 4px;
        }
        
        .chart-container {
            height: 20px;
            background: #27272a;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
        }
        
        .chart-correct {
            background: linear-gradient(90deg, #22c55e, #4ade80);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .chart-wrong {
            background: linear-gradient(90deg, #ef4444, #f87171);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .footer {
            margin-top: 40px;
            color: #52525b;
            font-size: 0.8rem;
        }
        
        .footer a {
            color: #60a5fa;
            text-decoration: none;
        }
        
        @media (max-width: 640px) {
            .container {
                flex-direction: column;
            }
            
            #canvas {
                width: 260px;
                height: 260px;
            }
        }
    </style>
</head>
<body>
    <h1>MNIST Digit Recognition</h1>
    <p class="subtitle">Draw a digit (0-9) and let the CNN predict it</p>
    
    <div class="container">
        <div class="card draw-section">
            <span class="canvas-label">Draw here</span>
            <canvas id="canvas" width="280" height="280"></canvas>
            <div class="buttons">
                <button class="btn-clear" onclick="clearCanvas()">Clear</button>
                <button class="btn-check" onclick="predict()">Check</button>
            </div>
        </div>
        
        <div class="card result-section">
            <h2>Prediction</h2>
            <div class="result-box">
                <div class="predicted-digit" id="prediction">-</div>
                <div class="confidence" id="confidence">Draw a digit and click Check</div>
            </div>
            
            <div class="feedback-section">
                <div class="feedback-label">Was this correct?</div>
                <div class="feedback-buttons">
                    <button class="btn-correct" onclick="feedback(true)">✓ Correct</button>
                    <button class="btn-wrong" onclick="feedback(false)">✗ Wrong</button>
                </div>
            </div>
            
            <div class="stats-section">
                <h2>Model Accuracy</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value correct" id="stat-correct">0</div>
                        <div class="stat-label">Correct</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value wrong" id="stat-wrong">0</div>
                        <div class="stat-label">Wrong</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value accuracy" id="stat-accuracy">0%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-correct" id="chart-correct" style="width: 0%"></div>
                    <div class="chart-wrong" id="chart-wrong" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>
    
    <p class="footer">
        Created by <strong>Bahman Farhadian</strong> • 
        CNN trained on MNIST • 
        <a href="https://github.com/bahman" target="_blank">GitHub</a>
    </p>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastPrediction = null;
        
        // Setup canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Drawing functions
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            e.preventDefault();
            
            const rect = canvas.getBoundingClientRect();
            let x, y;
            
            if (e.touches) {
                x = e.touches[0].clientX - rect.left;
                y = e.touches[0].clientY - rect.top;
            } else {
                x = e.clientX - rect.left;
                y = e.clientY - rect.top;
            }
            
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }
        
        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchmove', draw);
        
        function clearCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('prediction').textContent = '-';
            document.getElementById('confidence').textContent = 'Draw a digit and click Check';
            lastPrediction = null;
        }
        
        async function predict() {
            const imageData = canvas.toDataURL('image/png');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const result = await response.json();
                
                document.getElementById('prediction').textContent = result.digit;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${result.confidence.toFixed(1)}%`;
                
                lastPrediction = result.digit;
            } catch (error) {
                document.getElementById('confidence').textContent = 'Error: ' + error.message;
            }
        }
        
        async function feedback(isCorrect) {
            if (lastPrediction === null) {
                alert('Please make a prediction first!');
                return;
            }
            
            try {
                const response = await fetch('/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ correct: isCorrect })
                });
                
                const stats = await response.json();
                updateStats(stats);
                
                // Clear for next drawing
                clearCanvas();
            } catch (error) {
                console.error('Feedback error:', error);
            }
        }
        
        function updateStats(stats) {
            document.getElementById('stat-correct').textContent = stats.correct;
            document.getElementById('stat-wrong').textContent = stats.wrong;
            document.getElementById('stat-accuracy').textContent = stats.accuracy.toFixed(1) + '%';
            
            const total = stats.correct + stats.wrong;
            if (total > 0) {
                const correctPct = (stats.correct / total) * 100;
                const wrongPct = (stats.wrong / total) * 100;
                document.getElementById('chart-correct').style.width = correctPct + '%';
                document.getElementById('chart-wrong').style.width = wrongPct + '%';
            }
        }
        
        // Load initial stats
        fetch('/stats').then(r => r.json()).then(updateStats);
    </script>
</body>
</html>
"""


# =============================================================================
# API Routes
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    return HTML_TEMPLATE


@app.post("/predict")
async def predict(request: Request):
    """Predict digit from canvas image."""
    data = await request.json()
    image_data = data.get("image", "")
    
    if not image_data:
        return JSONResponse({"error": "No image data"}, status_code=400)
    
    try:
        # Preprocess and predict
        tensor = preprocess_image(image_data)
        
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            "digit": int(predicted.item()),
            "confidence": float(confidence.item() * 100)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/feedback")
async def feedback(request: Request):
    """Record user feedback on prediction."""
    data = await request.json()
    is_correct = data.get("correct", False)
    
    if is_correct:
        stats["correct"] += 1
    else:
        stats["wrong"] += 1
    
    total = stats["correct"] + stats["wrong"]
    accuracy = (stats["correct"] / total * 100) if total > 0 else 0
    
    return {
        "correct": stats["correct"],
        "wrong": stats["wrong"],
        "accuracy": accuracy
    }


@app.get("/stats")
async def get_stats():
    """Get current statistics."""
    total = stats["correct"] + stats["wrong"]
    accuracy = (stats["correct"] / total * 100) if total > 0 else 0
    
    return {
        "correct": stats["correct"],
        "wrong": stats["wrong"],
        "accuracy": accuracy
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("MNIST Digit Recognition Web Demo")
    print("By Bahman Farhadian")
    print("=" * 50)
    print()
    
    load_model()
    
    print()
    print("Starting server...")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
