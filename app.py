from flask import Flask,render_template,request
import cv2
from keras.models import Model ,load_model, model_from_json
import numpy as np
import Image_Captioning as ic
app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
@app.route('/')
def index():
    print("success")
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
def result():
    img=request.files['file1']
    img.save('static/file.jpg')
    pic=ic.encode_img('static/file.jpg')
    pic=pic.reshape((1,2048))
    final_caption=ic.predict_caption(pic)

    return render_template('result.html',data=final_caption)


if __name__=="__main__":
    app.run(debug=True)
