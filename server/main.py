from flask import Flask , request
from vqa_pretrained import demo
app = Flask(__name__)

@app.route("/sendImage" , methods=["GET"])
def sendImage():
    # img = request.get_json(force=True)["image"]
    # print (img)
    demo.main("./vqa_pretrained/test.jpg" , "Is there a train")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)