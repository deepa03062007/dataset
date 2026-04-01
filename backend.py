from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from googletrans import Translator

app = FastAPI()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("backend/final_model.keras", compile=False)
classes = ["Healthy","Diseased","Lumpy","Not Cow"]

users = []
cows = []

@app.get("/")
def home():
    return {"msg":"BovineBounty running"}

# LOGIN
@app.post("/login")
def login(data:dict):
    users.append(data)
    return {"msg":"Login success"}

# CHAT (MULTI LANGUAGE)
@app.post("/chat")
def chat(data:dict):
    msg = data["message"]

    t = translator.translate(msg, dest='en')
    text = t.text.lower()

    if "milk" in text:
        reply="Give green fodder and minerals"
    else:
        reply="Please explain problem clearly"

    final = translator.translate(reply, dest=t.src)
    return {"response":final.text}

# IMAGE AI
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    img = Image.open(io.BytesIO(await file.read())).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img)
    return {"prediction":classes[np.argmax(pred)]}

# PRODUCTS (50+)

@app.get("/products")
def products():
    return [
        {
            "name": "Cattle Ear Tags",
            "img": "https://m.media-amazon.com/images/I/61u6K9R1ZrL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Applicator Pliers",
            "img": "https://m.media-amazon.com/images/I/71pZQ8G9oNL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Hoof Paring Knife",
            "img": "https://m.media-amazon.com/images/I/61g2K9hF4PL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Hoof Nippers",
            "img": "https://m.media-amazon.com/images/I/71YpR9tZgHL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Hoof Rasp",
            "img": "https://m.media-amazon.com/images/I/61p+6h5X8bL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Electric Dehorner",
            "img": "https://m.media-amazon.com/images/I/61eU6h5X8bL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Curry Comb",
            "img": "https://m.media-amazon.com/images/I/71KXy2L7mKL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Livestock Shampoo",
            "img": "https://m.media-amazon.com/images/I/61MZ9K2wJzL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Mineral Lick Block",
            "img": "https://m.media-amazon.com/images/I/71u2K9R1ZrL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        },
        {
            "name": "Calf Feeding Bottle",
            "img": "https://m.media-amazon.com/images/I/61p9R2K1ZrL._SX679_.jpg",
            "link": "https://www.amazon.in/"
        }
    ]

# COW REGISTER
@app.post("/add_cow")
def add_cow(data:dict):
    cows.append(data)
    return {"msg":"added"}

@app.get("/get_cows")
def get_cows():
    return cows