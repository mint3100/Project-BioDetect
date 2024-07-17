import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps

# Project BioDetect
# Created : 24.07.16

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def predict_image(image):
    image = cv2.resize(image, (224, 224))
    normalized_image = (image.astype(np.float32) / 127.5) - 1
    input_data = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(input_data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        class_name, confidence_score = predict_image(image)
        display_image(file_path, class_name[2:], confidence_score)

def display_image(file_path, class_name, confidence_score):
    img = Image.open(file_path)
    img = img.resize((400, 400), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    
    panel.config(image=img)
    panel.image = img

    result_text.set(f"Class: {class_name}\nConfidence Score: {confidence_score}")

root = tk.Tk()
root.title("세포 판독기")

btn = tk.Button(root, text="Open Image", command=open_file)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 16))
result_label.pack()

root.mainloop()