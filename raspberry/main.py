import wiringpi as w
import os

btnLeft = 2
btnRight = 3
recieved = "test"

def setup():
    w.wiringPiSetupGpio()
    # Init pins 2,3 as input
    w.pinMode(btnLeft,0)
    w.pinMode(btnRight,0)

def loop():
    if not w.digitalRead(btnLeft):
        print ("left")
    if not w.digitalRead(btnRight):
        print ("right")
        os.system("espeak \" "+ recieved +" \" ")


if __name__ == '__main__':
    setup()
    while 1:
        loop()