import os
import torch
import torch.nn as nn
import numpy as np
import base64
from PIL import Image
from django.views import View
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0, densenet121
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'predictor', 'models')
DOCUMENTS_DIR = "C:/Users/Siddhartha/Documents"
FINGERPRINT_IMAGE = os.path.join(DOCUMENTS_DIR, "fingerprint.bmp")
LABELS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(device)


class ModelLoader:
    def __init__(self):
        self.resnet = self._load_resnet()
        self.efficientnet = self._load_efficientnet()
        self.densenet = self._load_densenet()
        self.xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_optuna_model.pkl"))

    def _load_resnet(self):
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet_model.pth"), map_location=device))
        return model.to(device).eval()

    def _load_efficientnet(self):
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "efficientnet_model.pth"), map_location=device))
        return model.to(device).eval()

    def _load_densenet(self):
        model = densenet121(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.Softmax(dim=1)
        )
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "densenet_model.pth"), map_location=device))
        return model.to(device).eval()


class PredictionPipeline:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.preprocessor = ImagePreprocessor()

    def run(self, image_path):
        tensor = self.preprocessor.preprocess(image_path)
        with torch.no_grad():
            p1 = self.model_loader.resnet(tensor).cpu().numpy()
            p2 = self.model_loader.efficientnet(tensor).cpu().numpy()
            p3 = self.model_loader.densenet(tensor).cpu().numpy()
        stacked = np.hstack([p1, p2, p3])
        probs = self.model_loader.xgb_model.predict_proba(stacked)[0]
        label_idx = np.argmax(probs)
        return LABELS[label_idx], probs


class PDFGenerator:
    @staticmethod
    def generate(context):
        template = get_template("predictor/report_template.html")
        html = template.render(context)
        response = HttpResponse(content_type="application/pdf")
        response["Content-Disposition"] = "inline; filename=report.pdf"
        pisa.CreatePDF(html, dest=response)
        return response


class HomeView(View):
    def get(self, request):
        return render(request, 'predictor/index.html')


class PredictView(View):
    def post(self, request):
        context = {}

        # ✅ Automatically load fingerprint.bmp from Documents
        if os.path.exists(FINGERPRINT_IMAGE):
            pipeline = PredictionPipeline()
            label, probs = pipeline.run(FINGERPRINT_IMAGE)

            # Read the image for display
            with open(FINGERPRINT_IMAGE, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            mime = "image/bmp"

            # Store prediction in session for PDF download
            request.session["last_prediction"] = {
                "predicted": label,
                "probs": probs.tolist(),
                "filename": "fingerprint.bmp",
                "image_data": f"data:{mime};base64,{encoded}"
            }

            context['result'] = label
            context['confidence'] = probs[np.argmax(probs)] * 100
            context['uploaded'] = "fingerprint.bmp"
        else:
            context['error'] = "No fingerprint.bmp found in the Documents folder."

        return render(request, 'predictor/index.html', context)


class DownloadReportView(View):
    def get(self, request):
        data = request.session.get("last_prediction")
        if not data:
            return HttpResponse("No report available.")

        zipped = zip(LABELS, data["probs"])
        context = {
            "predicted": data["predicted"],
            "filename": data["filename"],
            "zipped": zipped,
            "image_data": data["image_data"]
        }
        return PDFGenerator.generate(context)