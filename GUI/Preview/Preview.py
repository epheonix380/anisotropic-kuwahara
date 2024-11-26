import tkinter as tk
import numpy as np
from AnisotropicKuwahara.EXR import read 
from PIL import ImageTk, Image

class Preview(tk.Frame):
    def __init__ (self, parent):
        tk.Frame.__init__(self, parent)
        text_var = tk.StringVar()
        text_var.set("Hello, World!")
        self.parent = parent
        self.label = tk.Label(self, 
                              textvariable=text_var, 
                 anchor=tk.CENTER,       
                 height=4,              
                 width=6,                       
                 padx=15,               
                 pady=15,                  
                 underline=0,           
                 wraplength=250)
        self.label.pack(fill="both")

    def select_image(self, filepath):
        img:np.ndarray = read(input_path=filepath)*255
        height, width, _ = img.shape
        self.image = ImageTk.PhotoImage(image=Image.fromarray(img.astype(np.uint8)))
        self.label.config(image=self.image, text=None, height=height*2, width=width*2)
        self.label.update()
        self.update()

    