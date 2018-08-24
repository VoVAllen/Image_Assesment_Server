import io
import torch

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import models, transforms

from model import NIMA
from demo import bp

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import keras

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


# def generate_sentence(mean, std):
#     dict = {
#         (1, 1): '嘻嘻嘻，你是来搞笑的嘛，AI都欣赏不来这样的作品啦～',
#         (2, 1): '哼哼，要我说它美还差一点点哟～',
#         (3, 1): '这是让AI都叹为观止的美图！',
#         (1, 2): '噫，虽说不上是美图，但是还挺魔性的呢，哈哈哈～',
#         (2, 2): '你的作品还不赖呀，取景挺有眼光呢!',
#         (3, 2): '哇，你的作品颜值爆表，而且有很大的收藏价值哟！',
#     }
#     mean_level = 0
#     std_level = 0
#     if mean < 4: mean_level = 1
#     if 4 <= mean < 6: mean_level = 2
#     if 6 <= mean: mean_level = 3
#     if std < 1.7: std_level = 1
#     if std >= 1.7: std_level = 2
#     return dict[(mean_level, std_level)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


global graph
graph = tf.get_default_graph()

with tf.device('/GPU:0'):
    base_model = InceptionResNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('/data/server/weights/inception_res_v2_split.h5')


# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean


# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std


def get_score(img):
    img = img.resize((224, 224))
    x = img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    with graph.as_default():
        scores = model.predict(x, batch_size=1, verbose=0)[0]
        p_mean = mean_score(scores)
    return float(p_mean)


def get_dist(img):
    img = img.resize((224, 224))
    x = img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    with graph.as_default():
        scores = model.predict(x, batch_size=1, verbose=0)[0]
    np_dist = scores
    p_mean = mean_score(scores)
    p_std = std_score(scores)
    remap_score = remap(p_mean)
    result = {'dist': np_dist.tolist(), 'std': p_std, 'text': generate_sentence(p_mean, p_std),
              'score': "{:.2f}".format(remap_score),'original_score':"{:.2f}".format(p_mean)}
    return jsonify(result)


def remap(x):
    return min((x - 2) * 1.66, 10)


def generate_sentence(mean, std):
    mean = remap(mean)
    if std < 1.7:
        if mean < 3:
            return "嘻嘻嘻，你是来搞笑的嘛，AI都欣赏不来这样的作品啦～"
        if 3 <= mean < 5:
            return "哼哼，要我说它美还差一点点哟～"
        if 5 <= mean < 7:
            return "你的作品很棒棒哟，但是还有提升空间哟～"
        if mean > 7:
            return "这是让AI都叹为观止的美图！"
    if std >= 1.7:
        if mean < 4.5:
            return "噫，虽说不上是美图，但是还挺魔性的呢，哈哈哈～"
        if 4.5 <= mean < 5.5:
            return "你的作品还不赖呀，但是众说纷纭哦！"
        if 5.5 <= mean < 7:
            return "你的作品很棒棒哟，但是还有提升空间哟～"
        if mean > 7:
            return "哇，你的作品颜值爆表，而且有很大的收藏价值哟！"
    return "Error"



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
    app.run(host='0.0.0.0', port=5000, debug=False)
