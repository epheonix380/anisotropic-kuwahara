import tkinter as tk
import numpy as np
from AnisotropicKuwahara.EXR import read 
from PIL import ImageTk, Image
import math
import asyncio

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

    async def select_image(self, filepath):
        img:np.ndarray = read(input_path=filepath)*255
        await asyncio.sleep(0)
        height, width, _ = img.shape
        factor = 300/height
        newWidth = int(math.floor(width*factor))
        image = Image.fromarray(img.astype(np.uint8)).resize((newWidth, 300))
        await asyncio.sleep(0)
        self.image = ImageTk.PhotoImage(image=image)
        await asyncio.sleep(0)
        self.label.config(image=self.image, text=None, height=300, width=newWidth)
        self.label.update()
        self.update()

    