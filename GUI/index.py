import tkinter as tk
from .FileBar.FileBar import FileBar


class Index(tk.Frame):

    def __init__(self, parent: tk.Toplevel):
        tk.Frame.__init__(self, parent)
        parent.geometry("800x500")
        self.fileBar = FileBar(self)
        self.fileBar.grid(column=0, row=0)
        self.grid()
        
