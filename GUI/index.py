import tkinter as tk
import asyncio
from .FileBar.FileBar import FileBar
from .Preview.Preview import Preview


class Index(tk.Tk):

    def __init__(self, loop, interval=1/120):
        super().__init__()
        self.loop = loop
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.title("Kuwahara")
        self.geometry('800x600')
        self.wm_title("Kuwahara")
        self.preview = Preview(self)
        self.preview.grid(column=2, row=0, columnspan=4, rowspan=6)
        self.fileBar = FileBar(self)
        self.fileBar.grid(column=0, row=0, rowspan=10, columnspan=2)
        self.grid()
        self.tasks = []
        self.tasks.append(loop.create_task(self.updater(interval)))

    async def updater(self, interval):
        while True:
            self.update()
            await asyncio.sleep(interval)

    def close(self):
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        self.destroy()
        
