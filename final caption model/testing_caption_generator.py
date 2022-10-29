from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
from glob import glob
import json

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

max_length = 32 
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

img_folder = r"C:\Users\christine\Desktop\pics\*" # add \* at the back of the path
img_name = glob(img_folder +'*.jpg')
labels = {}

for i in img_name:
    filename = i[32:] # slice only the image name e.g. path="C:\Users\christine\Desktop\datamining\data\scamimage\05johnsmith1.jpg" we only want 05johnsmith1.jpg, pls slice accordingly 

    photo = extract_features(i, xception_model)
    img = Image.open(i)
    description = generate_desc(model, tokenizer, photo, max_length)

    #remove start and end
    description = description[6:-4]
    labels[filename] = description

json_object = json.dumps(labels) 

# change the filename to your name
with open('labels_chris.json', 'w') as f:
  json.dump(labels, f, ensure_ascii=False)

print('\n\n DONE ALR :DDD')