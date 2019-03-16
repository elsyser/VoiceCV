import speech_recognition as sr
import os

__all__ = [
    "transcribe",
]


def transcribe():
    recognizer = sr.Recognizer()
    
    print('-'*100)
    print("Listening...", '\n')
    with sr.Microphone() as src:
        audio = recognizer.listen(src)
        
    try:
        res = recognizer.recognize_google(audio).lower()
        print("You said: \"" + res + "\"")
        os.system("say \"" +res + "\"" )
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
    transcribe()
