import tkinter as tk
from tkinter.filedialog import askopenfilenames, askdirectory
from FileSystemWrapper.FileSystem import FileSystemWrapper

class FileBar(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.text = tk.StringVar()
        self.load_queue = []
        self.loaded_files = []
        self.search = tk.Entry(self, textvariable=self.text,validate="all", validatecommand=self.search_command)
        self.search.grid(row=0, column=0)
        self.add_file = tk.Button(self, text="Add File", command=self.select_file)
        self.add_file.grid(row=0, column=3 , padx=5)
        self.add_folder = tk.Button(self, text="Add Folder", command=self.select_folder)
        self.add_folder.grid(row=0, column=4, padx=5)

    def search_command(self):
        print(self.text.get())
        # Return True is needed for this function to work
        return True

    def select_folder(self):
        fs = FileSystemWrapper()
        foldername = askdirectory()
        fs.build_tree(root_path=foldername)
        fs.print_tree()

    def select_file(self):
        filename = askopenfilenames(defaultextension="exr", filetypes=[("EXR Files", "exr")])