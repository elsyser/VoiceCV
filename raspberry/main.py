# import packages
import wiringpi as w
import os
import requests
from PIL import Image
import select
import v4l2capture
import json
import base64
import speech_recognition as sr

# Inital state of variables and button pin declaration
response = None

btnLeft = 2
btnRight = 3

recieved = "test"

# VQA params
vqaServerIP = "10.106.1.157"
vqaServerPort = "5000"
vqaServerPath = "sendImage"

# Image capturing params
capServerIP = "10.106.1.157"
capServerPort = "5000"
capServerPath = "sendImage"

# Setup function for initialization
def setup():
    # Initializing buttons
    w.wiringPiSetupGpio()
    w.pinMode(btnLeft,0)
    w.pinMode(btnRight,0)

# Loop function (the main loop of the program)
def loop():
    if not w.digitalRead(btnLeft):
        # print ("left")
        imageCaptioning(getCameraImage())
    if not w.digitalRead(btnRight):
        # print ("right")
        visualQuestionAnswering(getCameraImage(),transcribe())
        
# Text-to-speach
def say(text):
    os.system("espeak \" "+ text +" \" ")


# Take a still frame and return it
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
    
# Send data in JSON format
def sendData(ip,port,path,payload):
    try:
        response = requests.post(
            "http://" + ip + ":" + port + "/" + path, data=json.dumps(payload)
        )
    except requests.exceptions.RequestException as e:
        print(e)


# Send data to server for image capturing and 'speak' response
def imageCaptioning(img):
    sendData(capServerIP,capServerPort,capServerPath,
        dict(
            tag = "captioning",
            imgData = base64.b64encode(img),
            question = ""
        )
    )
    # say(response.text)


# Send data to server for vqa and 'speak response
def visualQuestionAnswering(img, q):
    sendData(vqaServerIP,vqaServerPort,vqaServerPath,
        dict(
            tag = "vqa",
            imgData = base64.b64encode(img),
            question = q
        )
    )
    # say(response.text)


# Transcribe audio
def transcribe():
    recognizer = sr.Recognizer()
    
    print('-'*100)
    print("Listening...\n")
    # Use arecord to record short
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