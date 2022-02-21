# Breeds Finder
# Trackers
import pandas as pd
from flask import Flask, render_template, request, url_for, flash, redirect
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

import os
import numpy as np


names = {0: 'afghan_hound', 1: 'airedale', 2: 'american_staffordshire_terrier', 3: 'appenzeller', 4: 'basset', 5: 'beagle', 6: 'bedlington_terrier', 7: 'border_collie', 8: 'borzoi', 9: 'bouvier_des_flandres', 10: 'boxer', 11: 'brabancon_griffon', 12: 'briard', 13: 'brittany_spaniel', 14: 'cardigan', 15: 'chihuahua', 16: 'chow', 17: 'cocker_spaniel', 18: 'doberman', 19: 'english_foxhound', 20: 'english_springer', 21: 'flat_coated_retriever', 22: 'french_bulldog', 23: 'german_shepherd', 24: 'german_short_haired_pointer', 25: 'golden_retriever', 26: 'great_dane', 27: 'great_pyrenees', 28: 'greater_swiss_mountain_dog',
         29: 'groenendael', 30: 'ibizan_hound', 31: 'irish_setter', 32: 'italian_greyhound', 33: 'komondor', 34: 'labrador_retriever', 35: 'leonberg', 36: 'malinois', 37: 'miniature_pinscher', 38: 'miniature_schnauzer', 39: 'papillon', 40: 'pekinese', 41: 'pembroke', 42: 'pomeranian', 43: 'pug', 44: 'rottweiler', 45: 'saint_bernard', 46: 'saluki', 47: 'samoyed', 48: 'schipperke', 49: 'scottish_deerhound', 50: 'shetland_sheepdog', 51: 'shih_tzu', 52: 'siberian_husky', 53: 'sussex_spaniel', 54: 'toy_poodle', 55: 'toy_terrier', 56: 'vizsla', 57: 'weimaraner', 58: 'whippet', 59: 'yorkshire_terrier'}

UPLOAD_FOLDER = "static/uploads/"
app = Flask(__name__)
app.secret_key = "secret key"

MODEL_PATH = 'model/modeloInception.h5'

model = load_model(MODEL_PATH)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def identification(location):
    global key
    global probabilidad
    img = load_img(location, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img, [0])
    answer = np.argmax(model.predict(img), axis=-1)
    probability = round(np.max(model.predict(img)*100), 2)
    key = names[answer[0]]
    probabilidad = key.replace("_", " ").title()+ " con probabilidad de "+str(
        probability)+"%"
    print(key.replace("_", " ").title(), "con una ", probability, "%"
)
    return probabilidad


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('inicio.html')


@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        file_image = request.files['my_image']
        filename = secure_filename(file_image.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file_image.save(img_path)

        preds = identification(img_path)

        dfBreeds = pd.read_excel('static/info/info_razas.xlsx',
                                 header='infer', sheet_name='Hoja1', engine='openpyxl')
        key_final = key.replace("_", " ").title()
        resultado = {}

        def creaResultado(df):
            for i in df.index:
                if df['Raza'][i] == key_final:
                    resultado[df['Raza'][i]] = {
                        'Descripción': df['Descripción'][i],
                        'Origen': df['Origen'][i],
                        'Personalidad': df['Personalidad'][i],
                        'Salud': df['Salud'][i],
                        'Ejercicio': df['Ejercicio'][i],
                        'Nutrición': df['Nutrición'][i],
                        'Aseo': df['Aseo'][i]}
        creaResultado(dfBreeds)

        dfResultado = pd.DataFrame(
            [key for key in resultado.keys()], columns=['Raza'])
        dfResultado['Descripción'] = [value['Descripción']
                                      for value in resultado.values()]
        dfResultado['Origen'] = [value['Origen']
                                 for value in resultado.values()]
        dfResultado['Salud'] = [value['Salud'] for value in resultado.values()]
        dfResultado['Ejercicio'] = [value['Ejercicio']
                                    for value in resultado.values()]
        dfResultado['Nutrición'] = [value['Nutrición']
                                    for value in resultado.values()]
        dfResultado['Aseo'] = [value['Aseo'] for value in resultado.values()]

    return render_template("predict.html", prediction=preds, filename=filename, info=[dfResultado.to_html(header=True)])


@app.route('/display/<filename>')
def display_image(filename):

    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True, port=5005)
