import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import pyttsx3

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device1)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not found.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab image")
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(images=image, return_tensors="pt").to(device1)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    
    display_frame = frame.copy()
    cv2.putText(display_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Real-time Captioning", display_frame)

    # Speak the caption
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        print("Exiting...")
        break


camera.release()
cv2.destroyAllWindows()
