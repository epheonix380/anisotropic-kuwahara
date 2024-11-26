import tkinter as tk
from .FileBar.FileBar import FileBar
from .Preview.Preview import Preview


class Index(tk.Frame):

    def __init__(self, parent: tk.Toplevel):
        tk.Frame.__init__(self, parent)
        parent.geometry("800x500")
        self.preview = Preview(self)
        self.preview.grid(column=2, row=0, columnspan=12, rowspan=10)
        self.fileBar = FileBar(self)
        self.fileBar.grid(column=0, row=0, rowspan=10, columnspan=4)
        self.grid()
        
