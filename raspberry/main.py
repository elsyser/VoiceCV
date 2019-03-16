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

response = None

btnLeft = 2
btnRight = 3
recieved = "test"

vqaServerIP = "10.106.1.157"
vqaServerPort = "5000"
vqaServerPath = "sendImage"

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
    # image = Image.frombytes("RGB", (size_x, size_y), image_data)
    # image.save("image.jpg")
    video.close()

    return image_data
    

def sendData(ip,port,path,payload):
    try:
        response = requests.post(
            "http://" + ip + ":" + port + "/" + path, data=json.dumps(payload)
        )
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)


def imageCaptioning(img):
    sendData(vqaServerIP,vqaServerPort,vqaServerPath,
        dict(
            tag = "captioning",
            imgData = base64.b64encode(img),
            question = ""
        )
    )
    # print(response.json())

def visualQuestionAnswering(img, q):
    sendData(vqaServerIP,vqaServerPort,vqaServerPath,
        dict(
            tag = "vqa",
            imgData = base64.b64encode(img),
            question = q
        )
    )
    # print(response.json())

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