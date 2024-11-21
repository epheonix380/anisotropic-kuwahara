"""Proof of concept: integrate tkinter, asyncio and async iterator.

Terry Jan Reedy, 2016 July 25
"""

import asyncio
from random import randrange as rr
import tkinter as tk
from tkinter.filedialog import askopenfilename
from image_processor import async_image_processor


class App(tk.Tk):
    
    def __init__(self, loop, interval=1/120):
        super().__init__()
        self.loop = loop
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.title("Kuwahara")
        self.geometry('800x600')
        # Adding a title to the window
        self.wm_title("Test Application")

        # creating a frame and assigning it to container
        container = tk.Frame(self, height=400, width=600)
        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        # configuring the location of the container using grid
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # We will now create a dictionary of frames
        self.frames = {}
        # we'll create the frames themselves later but let's add the components to the dictionary.
        for F in (MainPage, SidePage, CompletionScreen):
            frame = F(container, self)

            # the windows class acts as the root window for the frames.
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Using a method to switch frames
        self.show_frame(MainPage)
        self.tasks = []
        self.images = ["D:/nyann/Downloads/Char.exr"]
        self.tasks.append(loop.create_task(self.updater(interval)))
        self.tasks.append(loop.create_task(self.image_processor(interval)))
    
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        # raises the current frame to the top
        frame.tkraise()

    async def image_processor(self, interval):
        while True:
            if len(self.images) > 0:
                first = self.images[0]
                self.images = self.images[1:] 
                print(first)               
                await async_image_processor(interval=interval, file_path=first)
                await asyncio.sleep(interval)
            else:
                self.image_processor()
                await asyncio.sleep(interval)

    async def updater(self, interval):
        while True:
            self.update()
            await asyncio.sleep(interval)

    def close(self):
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        self.destroy()

class MainPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page")
        label.pack(padx=10, pady=10)
        self.controller = controller

        another_button = tk.Button(
            self, 
            text="Select File",
            command=lambda: self.get_file()
        )
        # We use the switch_window_button in order to call the show_frame() method as a lambda function
        switch_window_button = tk.Button(
            self,
            text="Go to the Side Page",
            command=lambda: controller.show_frame(SidePage),
        )
        another_button.pack(side="top", fill=tk.X)
        switch_window_button.pack(side="bottom", fill=tk.X)
    
    def get_file(self):
        file_name = askopenfilename()
        print(file_name)
        self.controller.images += file_name



class SidePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="This is the Side Page")
        label.pack(padx=10, pady=10)

        switch_window_button = tk.Button(
            self,
            text="Go to the Completion Screen",
            command=lambda: controller.show_frame(CompletionScreen),
        )
        switch_window_button.pack(side="bottom", fill=tk.X)


class CompletionScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Completion Screen, we did it!")
        label.pack(padx=10, pady=10)
        switch_window_button = tk.Button(
            self, text="Return to menu", command=lambda: controller.show_frame(MainPage)
        )
        switch_window_button.pack(side="bottom", fill=tk.X)

loop = asyncio.get_event_loop()
app = App(loop)
loop.run_forever()
loop.close()