from GUI.index import Index
import tkinter as tk

from async_tkinter_loop import async_handler, async_mainloop

if __name__ == "__main__":
    root = tk.Tk()
    app = Index(root)
    async_mainloop(root=root)