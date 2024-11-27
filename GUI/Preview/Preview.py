import tkinter as tk
import numpy as np
from AnisotropicKuwahara.EXR import read 
from AnisotropicKuwahara.Kuwahara import Kuwahara
from PIL import ImageTk, Image
import math

class Preview(tk.Frame):
    def __init__ (self, parent):
        tk.Frame.__init__(self, parent)
        text_var = tk.StringVar()
        text_var.set("Preview")
        self.parent = parent
        self.label = tk.Label(self, 
                              textvariable=text_var, 
                 anchor=tk.NE,       
                 height=4,              
                 width=6,                       
                 padx=15,               
                 pady=15,                  
                 underline=0,           
                 wraplength=250)
        self.label.pack(fill="both", anchor=tk.NE)

    def _async_image_processor(self):
        kuwahara = Kuwahara()
        img = kuwahara.process(self.pre_image)*255
        image = Image.fromarray(self.pre_image.astype(np.uint8)).resize((self.width, self.height))
        self.image = ImageTk.PhotoImage(image=image)
        self._async_image_updater()

    def _async_image_updater(self):
        self.label.config(image=self.image, text=None, height=self.height, width=self.width)
        self.label.update()
        self.update()

    def _async_image_loader(self, filepath):
        img:np.ndarray = read(input_path=filepath)*255
        height, width, _ = img.shape
        factor = 300/height
        newWidth = int(math.floor(width*factor))
        self.width = newWidth
        self.height = 300
        self.pre_image = img
        self._async_image_processor()
        image = Image.fromarray(img.astype(np.uint8)).resize((newWidth, 300))
        self.pre_image = ImageTk.PhotoImage(image=image)

    def select_image(self, filepath):
        self._async_image_loader(filepath=filepath)

    