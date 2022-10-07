# -*- coding: utf-8 -*-
# @Time    : 2022-05-16 13.31
# @Author  : Nuwan Madhusanka
# @Email   : nuwan@xdoto.io
# @File    : draw_results.py
# @Software: PyCharm

import pickle
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename

global loaded_model

def history_(history):
    print(history.keys())

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def choose_model():
    global loaded_model,l_selected_model
    filename = askopenfilename()
    print(filename)
    l_selected_model.config(text=filename)
    loaded_model = pickle.load(open(filename, 'rb'))

def predict():
    loaded_model

with open(r"../basic/models/trainHistoryDict", "rb") as data_file:
    data = pickle.load(data_file)

def choose_image():
    global image,c_image,c_image_container
    filename = askopenfilename()
    print(filename)

    image = PhotoImage(file=filename)
    c_image.itemconfig(c_image_container,image = image)


master = Tk()
master.title("Fabric Defect detection - Prediction system - [MSc in AI - 12 batch]")
master.geometry("850x800")

width = 150
master.grid_columnconfigure(0, minsize=50)
master.grid_columnconfigure(1, minsize=width)
master.grid_columnconfigure(2, minsize=width)
master.grid_columnconfigure(3, minsize=width)
master.grid_columnconfigure(4, minsize=width)
master.grid_columnconfigure(5, minsize=width)
master.grid_columnconfigure(6, minsize=50)

master.rowconfigure((0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), weight=2)

l_title = Label(master,text="Fabric Defet Detection - Model Testing" ,font="arial 20 bold")
b_select_models = Button(master, text="Select Model", width=13,command=choose_model, font="arial 10 bold")
l_selected_model = Label(master ,font="arial 10")

b_select_image = Button(master, text="Select Image", width=13,command=choose_image, font="arial 10 bold")
c_image = Canvas(master, width = 500, height = 500)
img = PhotoImage(file="/home/nuwan/Applications/AI_cource/research/programing/keras/UI/images/processed/Adefault.png")
c_image_container = c_image.create_image(20, 20, anchor=NW,image=img)

b_predict = Button(master, text="Predict", width=13,command=predict, font="arial 10 bold")
l_prediction = Label(master ,font="arial 10")

l_title.grid(row=1, column=1, columnspan=4, pady=7)
b_select_models.grid(row=3, column=1, pady=7)
b_select_image.grid(row=5, column=1, pady=7)
l_selected_model.grid(row=3, column=2, pady=7)
c_image.grid(row=5, column=2, pady=7,columnspan=3, rowspan=9)
master.mainloop()