# import picamera
# import picamera.array
import wiringpi as w
import os
import requests
from PIL import Image
import select
import v4l2capture
import json
import base64
import speech_recognition as sr


btnLeft = 2
btnRight = 3
recieved = "test"

# Setup function for initialization
def setup():
    # Initializing buttons
    w.wiringPiSetupGpio()
    w.pinMode(btnLeft,0)
    w.pinMode(btnRight,0)

# Loop function (the main loop of the program)
def loop():
    if not w.digitalRead(btnLeft):
        print ("left")
    if not w.digitalRead(btnRight):
        print ("right")
        say(recieved)
        # print(getCameraImage())
        visualQuestionAnswering(getCameraImage(),transcribe())
        

def say(text):
    os.system("espeak \" "+ text +" \" ")

def getCameraImage():
    video = v4l2capture.Video_device("/dev/video0")
    size_x, size_y = video.set_format(320, 240)
    video.create_buffers(1)
    video.queue_all_buffers()
    video.start()
    select.select((video,), (), ())
    image_data = video.read()
    video.close()
    image = Image.frombytes("RGB", (size_x, size_y), image_data)
    image.save("image.jpg")
    return image_data
    

def sendData(ip,port,path,payload):
    r = requests.post(
        "http://" + ip + ":" + port + "/" + path, data=json.dumps(payload)
    )


# def imageCaptioning():
#     r = requests.post(
#         "https://api.deepai.org/api/densecap",
#         files={
#             'image': open('./image.jpg', 'rb'),
#         },
#         headers={'api-key': 'YOUR_API_KEY'}
#     )
#     print(r.json())

def visualQuestionAnswering(img, q):
    sendData("10.106.1.157","5000","sendImage",
        dict(
            tag = "vqa",
            imgData = base64.b64encode(img),
            question = q
        )
    )

def transcribe():
    recognizer = sr.Recognizer()
    
    print('-'*100)
    print("Listening...\n")
    # with sr.Microphone() as src:
    #     audio = recognizer.listen(src)
    os.system("arecord -D plughw:1,0 -d 5 temp.wav")
    with sr.WavFile("temp.wav") as source:
        audio = recognizer.record(source)  
        print("Working")
        
    try:
        res = recognizer.recognize_google(audio).lower()
        print("You said: \"" + res + "\"")
        print('-'*100)
        return res
    
    except sr.UnknownValueError:
        print("Could not understand audio")
        print('-'*100)
        return None
        
    except sr.RequestError as e:
        print("Could not request results from the service; {0}".format(e))
        print('-'*100)
        return None

if __name__ == '__main__':
    setup()
    while 1:
        loop()