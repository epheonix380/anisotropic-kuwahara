import tkinter as tk
import numpy as np
from AnisotropicKuwahara.EXR import read 
from AnisotropicKuwahara.Kuwahara import Kuwahara
from ShadyKuwahara.render import renderFromNDarray
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
        vcmd = (self.register(self.callback))
        self.radius = 5
        self.radius_text = tk.Entry(self, validate='all', validatecommand=(vcmd, '%P')) 
        self.radius_text.insert(0, "5")
        self.radius_text.pack(anchor=tk.S)

    def callback(self, P):
        if str.isdigit(P) or P == "":
            try:
                rad = int(P)
            except:
                rad = 5
            if rad > 100:
                print("100 is cap")
            else:
                self.radius = rad
                asyncio.ensure_future(self._async_image_processor(), loop=self.parent.loop)
            return True
        else:
            return False

    async def _async_image_processor(self):
        img = renderFromNDarray(self.pre_image, self.radius)
        await asyncio.sleep(1/120)
        image = Image.fromarray(img.astype(np.uint8)).resize((self.width, self.height))
        self.image = ImageTk.PhotoImage(image=image)
        await asyncio.sleep(1/120)
        await self._async_image_updater()

    async def _async_image_updater(self):
        self.label.config(image=self.image, text=None, height=self.height, width=self.width)
        self.label.update()
        self.update()

    async def _async_image_loader(self, filepath):
        img:np.ndarray = read(input_path=filepath)
        height, width, _ = img.shape
        factor = 300/height
        newWidth = int(math.floor(width*factor))
        self.width = newWidth
        self.height = 300
        self.pre_image = img
        await self._async_image_processor()
        #image = Image.fromarray(img.astype(np.uint8)).resize((newWidth, 300))
        #self.pre_image = ImageTk.PhotoImage(image=image)

    def select_image(self, filepath):
        asyncio.ensure_future(self._async_image_loader(filepath=filepath), loop=self.parent.loop)

    