# import picamera
# import picamera.array
import wiringpi as w
import os
import requests
from PIL import Image
import select
import v4l2capture


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
        
        visualQuestionAnswering(getCameraImage(),"Testing?")
        

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
    return image_data
    # image = Image.frombytes("RGB", (size_x, size_y), image_data)
    # image.save("image.jpg")

def sendData(ip,port,path,data):
    r = requests.get(
        "http://" + ip + ":" + port + "/" + path, data
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
    sendData("localhost","5000","",
        dict(
            tag = "vqa",
            imgData = img,
            question = q
        )
    )

if __name__ == '__main__':
    setup()
    while 1:
        loop()