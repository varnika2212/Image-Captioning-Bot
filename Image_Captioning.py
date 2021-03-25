from keras.models import Model,load_model, model_from_json
import numpy as np
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.resnet50  import ResNet50, preprocess_input,decode_predictions
import PIL

word_to_idx={}
idx_to_word={}
total_words= [line.strip() for line in open("total_words.txt", 'r')]
for i,word in enumerate(total_words):
    word_to_idx[word]=i+1
    idx_to_word[i+1]=word

idx_to_word[1846]="startseq"
word_to_idx["startseq"]=1846
word_to_idx["endseq"]=1847
idx_to_word[1847]="endseq"

vocab_size=len(word_to_idx)+1

json_file = open('my_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
 # load weights into new model
loaded_model.load_weights("wt_model.h5")

max_len=35

model=ResNet50(weights="imagenet",input_shape=(224,224,3))
model_new=Model(model.input,model.layers[-2].output)
def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    #normalisation
    img=preprocess_input(img)
    return img

def encode_img(img):
    img=preprocess_img(img)
    feature_vector=model_new.predict(img)
    feature_vector=feature_vector.reshape((2048,))
    #print(feature_vector.shape)
    return feature_vector

def predict_caption(photo):
    in_text="startseq"
    for i in range(max_len):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence],maxlen=max_len,padding='post')
        ypred=loaded_model.predict([photo,sequence])
        ypred=ypred.argmax() #word with max prob always - greedy sampling
        word=idx_to_word[ypred]
        in_text +=(' '+ word)

        if word=="endseq":
            break

    final_caption=in_text.split()[1:15]
    final_caption=' '.join(final_caption)

    return final_caption
