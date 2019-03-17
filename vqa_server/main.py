from flask import Flask , request, make_response
import base64
import os
from PIL import Image
from vqa_pretrained import vqa

# from vqa_pretrained import vqa
app = Flask(__name__)

# http://url:5000/sendImage
# data = {
#   tag: {string},
#   imgData: {base64 string},
#   question: {string}
# }
@app.route("/sendImage" , methods=["POST"])
def sendImage():
    #Parse the request to json
    req = request.get_json(force=True)
    #Decode base64 to image file
    img = base64.b64decode(req["imgData"])   
    image = Image.frombytes("RGB", (320, 240), img)
    #Save the image to file system
    image.save("image.jpg")
    question = req["question"]
    print "Question is " + question
    os.system("cd vqa_pretrained; python vqa.py -image_file_name ../image.jpg -question \"" + question +"\"")
    res = ''
    # Read the outputed file of vqa.py and send it as response
    with open("vqa_pretrained/output.txt" , 'r') as f:
        res = f.read()
    print res
    return make_response(res , 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)