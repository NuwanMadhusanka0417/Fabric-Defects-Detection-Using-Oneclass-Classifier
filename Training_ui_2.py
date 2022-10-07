# -*- coding: utf-8 -*-
# @Time    : 2022-04-20 17.07
# @Author  : Nuwan Madhusanka
# @Email   : nuwan@xdoto.io
# @File    : ui.py.py
# @Software: PyCharm
import math
import os
from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import numpy as np
dataset = "tile"
r = 0.5
g = 0.85

command = "python3 main_script.py -d "+ dataset + " -g " + str(g) + " -r " + str(r)
print(command)

# print(os.system("python3 main_script.py -d tile -g 0.5 -r 0.05"))

def print_cbox(event):
    country = event.widget.get()
    print(country)

def cbox():
    print(datasetchoosen.current(), datasetchoosen.get())
    print("grad ",e_grad.get())
    print("ramp ",e_ramp.get())
    os.system("python3 main_script.py -d "+str(datasetchoosen.get())+" -g "+str(e_grad.get())+" -r "+str(e_ramp.get()))
    print("python3 main_script.py -d "+str(datasetchoosen.get())+" -g "+str(e_grad.get())+" -r "+str(e_ramp.get()))
def load_images():
    global img,Carpet_img_n,Carpet_img_d,UMDAA_img_d,UMDAA_img_n,capsulle_img_d,capsulle_img_n,cable_img_d,cable_img_n
    global hazelnut_img_d,hazelnut_img_n,leather_img_d,leather_img_n,tile_img_d,tile_img_n,transistor_img_d,transistor_img_n

    img = PhotoImage(file="images/processed/Adefault.png")

    Carpet_img_d = PhotoImage(file="images/processed/Acarpet_d.png")
    Carpet_img_n = PhotoImage(file="images/processed/Acarpet_n.png")

    UMDAA_img_d = PhotoImage(file="images/processed/uaamd_d.png")
    UMDAA_img_n = PhotoImage(file="images/processed/uaamd_n.png")

    capsulle_img_d = PhotoImage(file="images/processed/Acapsule_d.png")
    capsulle_img_n = PhotoImage(file="images/processed/Acapsule_n.png")

    cable_img_d = PhotoImage(file="images/processed/Acable_d.png")
    cable_img_n = PhotoImage(file="images/processed/Acable_n.png")

    hazelnut_img_d = PhotoImage(file="images/processed/Ahazelnut_d.png")
    hazelnut_img_n = PhotoImage(file="images/processed/Ahazelnut_n.png")

    leather_img_d = PhotoImage(file="images/processed/Aleather_d.png")
    leather_img_n = PhotoImage(file="images/processed/Aleather_n.png")

    tile_img_d = PhotoImage(file="images/processed/Atile_d.png")
    tile_img_n = PhotoImage(file="images/processed/Atile_n.png")

    transistor_img_d = PhotoImage(file="images/processed/Atransistor_d.png")
    transistor_img_n = PhotoImage(file="images/processed/Atransistor_n.png")

def TextBoxUpdate(event):
    print(datasetchoosen.current(), datasetchoosen.get())
    if datasetchoosen.get() == ' Carpet':
        l_dataset_details.config(text="Num of Images = 400  [Positive Images - 308]\n\t\t      [Negetive Images - 92]\nResolution = 1024x1024")
        c_defect_image.itemconfig(c_d_container,image = Carpet_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = Carpet_img_n)
    elif datasetchoosen.get() == ' UMDAA':
        l_dataset_details.config(text="Num of Images = 32135  [Positive Images - 23196]\n\t\t      [Negetive Images - 8939]\nResolution = 960x1280")
        c_defect_image.itemconfig(c_d_container,image = UMDAA_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = UMDAA_img_n)
    elif datasetchoosen.get() == ' Capsulle':
        l_dataset_details.config(text="Num of Images = 351  [Positive Images - 242]\n\t\t      [Negetive Images - 109]\nResolution = 1000x1000")
        c_defect_image.itemconfig(c_d_container,image = capsulle_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = capsulle_img_n)
    elif datasetchoosen.get() == ' Cable':
        l_dataset_details.config(text="Num of Images = 374  [Positive Images - 282]\n\t\t      [Negetive Images - 92]\nResolution = 1024x1024")
        c_defect_image.itemconfig(c_d_container,image = cable_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = cable_img_n)
    elif datasetchoosen.get() == ' Hazelnut':
        l_dataset_details.config(text="Num of Images = 401  [Positive Images - 331]\n\t\t      [Negetive Images - 70]\nResolution = 1024x1024")
        c_defect_image.itemconfig(c_d_container,image = hazelnut_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = hazelnut_img_n)
    elif datasetchoosen.get() == ' Leather':
        l_dataset_details.config(text="Num of Images = 378  [Positive Images - 286]\n\t\t      [Negetive Images - 92]\nResolution = 1024x1024")
        c_defect_image.itemconfig(c_d_container,image = leather_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = leather_img_n)
    elif datasetchoosen.get() == ' Tile':
        l_dataset_details.config(text="Num of Images = 347  [Positive Images - 263]\n\t\t      [Negetive Images - 84]\nResolution = 840x840")
        c_defect_image.itemconfig(c_d_container,image = tile_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = tile_img_n)
    elif datasetchoosen.get() == ' Transistor':
        l_dataset_details.config(text="Num of Images = 313  [Positive Images - 273]\n\t\t      [Negetive Images - 40]\nResolution = 1024x1024")
        c_defect_image.itemconfig(c_d_container,image = transistor_img_d)
        c_nondefect_image.itemconfig(c_n_container,image = transistor_img_n)

def exit():
    master.quit()

def change_graph_grad_sv(sv):
    print(sv.get())
    if sv.get():
        grad_val = float(sv.get())
    else:
        grad_val = 0

    if e_ramp.get():
        ramp_val = float(e_ramp.get())
    else:
        ramp_val = 0
    print("ramp val = ",ramp_val)
    # z = [(i ** 2) * 1.1 for i in range(101)]
    inner = [-1*i*grad_val if i < 0 else 0 for i in x]
    z = [min(i,ramp_val) for i in inner]

    graph_plot1.clear()
    graph_plot1.plot(x, y)
    graph_plot1.plot(x,z)
    c_graph.draw()

def change_graph_ramp_sv(sv):
    print(sv.get())
    if sv.get():
        grad_val = float(e_grad.get())
    else:
        grad_val = 0

    if e_ramp.get():
        ramp_val = float(sv.get())
    else:
        ramp_val = 0
    print("ramp val = ",ramp_val)
    # z = [(i ** 2) * 1.1 for i in range(101)]
    inner = [-1*i*grad_val if i < 0 else 0 for i in x]
    z = [min(i,ramp_val) for i in inner]

    graph_plot1.clear()
    graph_plot1.plot(x, y)
    graph_plot1.plot(x,z)
    c_graph.draw()

def e_robust_val_graph_eta_sv(sv):
    print(sv.get())
    if sv.get():
        eta_val = float(sv.get())
    else:
        eta_val = 0

    beta_val = 1 / (1 - math.exp(-1 * eta_val))

    y = [beta_val * (1 - math.exp(-1 * eta_val * (-1*i))) if i < 0 else 0 for i in x]

    graph_plot1_robust.clear()
    graph_plot1_robust.plot(x, y)

    c_graph_robust.draw()

def e_ramp_val_graph_sv(sv):
    print(sv.get())
    if sv.get():
        ramp_val = float(e_ramp_graph.get())
    else:
        ramp_val = 0

    graph_plot1_Ramp.clear()
    x = np.linspace(-2, 2, 200)
    y = [min(ramp_val,-i) if i < 0 else 0 for i in x]

    graph_plot1_Ramp.plot(x, y)

    c_graph_Ramp.draw()

master = Tk()
master.title("Fabric Defect detection - Training system - [MSc in AI - 12 batch]")
master.geometry("1850x1020")
master.grid_columnconfigure(0, minsize=50)
master.grid_columnconfigure(1, minsize=200)
master.grid_columnconfigure(2, minsize=200)
master.grid_columnconfigure(3, minsize=10)
master.grid_columnconfigure(4, minsize=200)
master.grid_columnconfigure(5, minsize=200)
master.grid_columnconfigure(6, minsize=50)

master.grid_columnconfigure(7, minsize=200)
master.grid_columnconfigure(8, minsize=200)
master.grid_columnconfigure(9, minsize=90)
master.grid_columnconfigure(10, minsize=200)
master.grid_columnconfigure(11, minsize=200)
master.grid_columnconfigure(12, minsize=50)

master.rowconfigure((0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21), weight=2)
master.rowconfigure(4, weight=1)
master.rowconfigure(3, weight=3)
master.resizable(width=False, height=False)

l_title = Label(master,text="Fabric Defet Detection - Model Training" ,font="arial 20 bold")

l_Dataset = Label(master,text="Select Dataset" ,font="arial 12 bold")

l_dataset_details = Label(master,text="             \n           ",font="arial 13 italic bold",justify=LEFT)
n = StringVar()
datasetchoosen = ttk.Combobox(master, width=10,
                              textvariable=n, validatecommand=cbox)

# Adding combobox drop down list
datasetchoosen['values'] = (' Carpet',
                          ' UMDAA',
                          ' Capsulle',
                          ' Cable',
                          ' Hazelnut',
                          ' Leather',
                          ' Tile',
                          ' Transistor')

datasetchoosen.bind("<<ComboboxSelected>>", TextBoxUpdate)
# monthchoosen.current(0)

l_positive_image = Label(master,text="Negative Image" ,font="arial 10",justify=LEFT)
l_negative_image = Label(master,text="Positive Image" ,font="arial 10",justify=LEFT)

c_defect_image = Canvas(master, width = 300, height = 300)
c_nondefect_image = Canvas(master, width = 300, height = 300)
load_images()
c_d_container = c_defect_image.create_image(20, 20, anchor=NW, image=img)
c_n_container = c_nondefect_image.create_image(20, 20, anchor=NW, image=img)


l_Grad_Ratio = Label(master,text="Gradient Ratio" ,font="arial 12 bold")
l_Ramp_Limit = Label(master,text="Ramp Limit" ,font="arial 12 bold")

e_grad_val = StringVar()
e_grad_val.trace("w", lambda name, index, mode, sv=e_grad_val: change_graph_grad_sv(sv))
e_grad = Entry(master, width=10, font=("arial 12 bold"), justify=RIGHT, textvariable=e_grad_val)

e_ramp_val = StringVar()
e_ramp_val.trace("w", lambda name, index, mode, sv=e_ramp_val: change_graph_ramp_sv(sv))
e_ramp = Entry(master, width=10, font=("arial 12 bold"), justify=RIGHT, textvariable=e_ramp_val)

graph_fig = Figure(figsize=(5, 2),dpi=120)
graph_plot1 = graph_fig.add_subplot(111)
# graph_plot2 = graph_fig.add_subplot(111)

c_graph = FigureCanvasTkAgg(graph_fig, master=master)

x = np.linspace(-2,2,200)
y = [-i if i<0 else 0 for i in x]
graph_plot1.plot(x,y)

c_graph.draw()

b_train = Button(master, text="Train", width=13,command=cbox, font="arial 10 bold")
b_exit = Button(master, text="Exit", width=13,command=exit, font="arial 10 bold")

l_title.grid(row=1, column=3, columnspan=7, pady=7)
l_Dataset.grid(row=3, column=1, pady=7)

l_dataset_details.grid(row=3, column=4,columnspan=3,rowspan=2, pady=7)
datasetchoosen.grid(row=3, column=2)

l_positive_image.grid(row=5, column=1, pady=7)
l_negative_image.grid(row=5, column=4, pady=7)

c_defect_image.grid(row=6, column=1, columnspan=2,rowspan=6,pady=7)
c_nondefect_image.grid(row=6, column=4,columnspan=2,rowspan=6, pady=7)

l_Grad_Ratio.grid(row=13, column=1, pady=7)
l_Ramp_Limit.grid(row=13, column=4, pady=7)

e_grad.grid(row=13, column=2, pady=7)
e_ramp.grid(row=13, column=5, pady=7)

c_graph.get_tk_widget().grid(row=15, column=1, columnspan=5,rowspan=5, pady=7)

b_train.grid(row=20, column=4, pady=7)
b_exit.grid(row=20, column=5, pady=7)

#------------------Right Side--------------------------#
l_exist_loss = Label(master,text="Existing Loss Functions" ,font="arial 15 bold",)
l_hinge_loss = Label(master,text="Hinge Loss Functions" ,font="arial 12 bold",anchor='e')
l_Ramp_loss = Label(master,text="Ramp Loss Functions" ,font="arial 12 bold",anchor='w')
l_Robust_loss = Label(master,text="Robust Loss Functions" ,font="arial 12 bold",anchor='w')

graph_fig_hinge = Figure(figsize=(5, 2),dpi=120)
graph_plot1_hinge = graph_fig_hinge.add_subplot(111)
c_graph_hinge = FigureCanvasTkAgg(graph_fig_hinge, master=master)
x = np.linspace(-2,2,200)
y = [-i if i<0 else 0 for i in x]
graph_plot1_hinge.plot(x,y)
c_graph_hinge.draw()

# Ramp graph
l_ramp_graph = Label(master,text="Ramp Limit" ,font="arial 12")

e_ramp_val_graph = StringVar()
e_ramp_val_graph.trace("w", lambda name, index, mode, sv=e_ramp_val_graph: e_ramp_val_graph_sv(sv))
e_ramp_graph = Entry(master, width=10, font=("arial 12 bold"), justify=RIGHT, textvariable=e_ramp_val_graph)

graph_fig_Ramp = Figure(figsize=(5, 2),dpi=120)
graph_plot1_Ramp = graph_fig_Ramp.add_subplot(111)
c_graph_Ramp = FigureCanvasTkAgg(graph_fig_Ramp, master=master)
c_graph_Ramp.draw()

# Robust graph
l_robust_graph_eta = Label(master,text="Eta value" ,font="arial 12")

e_robust_val_graph_eta = StringVar()
e_robust_val_graph_eta.trace("w", lambda name, index, mode, sv=e_robust_val_graph_eta: e_robust_val_graph_eta_sv(sv))
e_robust_graph_eta = Entry(master, width=10, font=("arial 12 bold"), justify=RIGHT, textvariable=e_robust_val_graph_eta)

graph_fig_robust = Figure(figsize=(5, 2),dpi=120,)

graph_plot1_robust = graph_fig_robust.add_subplot(111)
c_graph_robust = FigureCanvasTkAgg(graph_fig_robust, master=master)

c_graph_robust.draw()

l_exist_loss.grid(row=3, column=7,columnspan=3, pady=7)
l_hinge_loss.grid(row=5, column=7,columnspan=2, pady=7)
l_Ramp_loss.grid(row=10, column=7,columnspan=2, pady=7)
l_Robust_loss.grid(row=15, column=7,columnspan=2, pady=7)

# hinge
c_graph_hinge.get_tk_widget().grid(row=5, column=9, columnspan=3,rowspan=4, pady=7)

#ramp
l_ramp_graph.grid(row=12, column=7, pady=7)
e_ramp_graph.grid(row=12, column=8, pady=7)
c_graph_Ramp.get_tk_widget().grid(row=10, column=9, columnspan=3,rowspan=4, pady=7)

#robust
l_robust_graph_eta.grid(row=17, column=7, pady=7)

e_robust_graph_eta.grid(row=17, column=8, pady=7)

c_graph_robust.get_tk_widget().grid(row=15, column=9, columnspan=3,rowspan=4, pady=7)

master.mainloop()