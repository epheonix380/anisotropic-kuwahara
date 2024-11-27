import tkinter as tk
from .FileBar.FileBar import FileBar
from .Preview.Preview import Preview


class Index(tk.Frame):

    def __init__ (self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.preview = Preview(self)
        self.preview.grid(column=2, row=0, columnspan=4, rowspan=6)
        self.fileBar = FileBar(self)
        self.fileBar.grid(column=0, row=0, rowspan=10, columnspan=2)
        self.grid()

        
