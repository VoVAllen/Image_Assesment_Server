import io
import torch

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import models, transforms

from model import NIMA
from demo import bp

app = Flask(__name__)
app.register_blueprint(bp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load("/data/server/weights/nima.pth"))
model = model.to(device)
model.eval()

val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_score(img):
    new_img = val_transform(img).unsqueeze(0).to(device)
    outputs = model(new_img)
    dist = torch.arange(1, 11).float().to(device)
    p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
    return float(p_mean)


def get_dist(img):
    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    new_img = val_transform(img).unsqueeze(0).to(device)
    outputs = model(new_img)
    dist = torch.arange(1, 11).float().to(device)
    p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
    np_dist = outputs.data.cpu().numpy()
    result = {'dist': np_dist.tolist(), 'var': np.var(np_dist).tolist()}
    return jsonify(result)


def remap(x):
    return sigmoid(x - 4.5) * 10


@app.route("/get_score", methods=['POST'])
def score_route():
    f = request.files['file']
    # img_np=file_to_numpy(f)

    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    img = Image.open(in_memory_file).convert("RGB")
    remap_score = remap(get_score(img))
    return "{:.2f}".format(remap_score)


@app.route("/get_dist", methods=['POST'])
def dist_route():
    f = request.files['file']
    # img_np=file_to_numpy(f)

    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    img = Image.open(in_memory_file).convert("RGB")
    return get_dist(img)


@app.route("/")
def good():
    return "good"


def file_to_numpy(f):
    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    return np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
