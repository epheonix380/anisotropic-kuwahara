import tkinter as tk
from tkinter.filedialog import askopenfilenames, askdirectory
from FileSystemWrapper.FileSystem import FileSystemWrapper, File

class FileBar(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.text = tk.StringVar()
        self.load_queue = []
        self.list = tk.Listbox(self)
        self.fileList = []
        self.list.bind("<<ListboxSelect>>", self.select_from_list)
        self.list.grid(row=2, column=0, rowspan=8, columnspan=2)
        self.search = tk.Entry(self, textvariable=self.text,validate="all", validatecommand=self.search_command)
        self.search.grid(row=0, column=0, columnspan=2)
        self.add_file = tk.Button(self, text="Add File", command=self.select_file)
        self.add_file.grid(row=1, column=0 , padx=5)
        self.add_folder = tk.Button(self, text="Add Folder", command=self.select_folder)
        self.add_folder.grid(row=1, column=1, padx=5)

    def select_from_list(self, item):
        print(self.fileList[self.list.curselection()[0]].path)
        self.parent.preview.select_image(self.fileList[self.list.curselection()[0]].path)

    def search_command(self):
        print(self.text.get())
        # Return True is needed for this function to work
        return True

    def select_folder(self):
        fs = FileSystemWrapper()
        foldername = askdirectory()
        fs.build_tree(root_path=foldername)
        files = fs.find_all_with_ext("exr")
        self.load_queue = self.load_queue + files
        print(files)
        self.empty_load_queue()

    def select_file(self):
        filename = askopenfilenames(defaultextension="exr", filetypes=[("EXR Files", "exr")])
        path = filename[0]
        name = path.split("/")[-1]
        self.load_queue.append(File(name=name, path=path, size=3))
        self.empty_load_queue()


    def empty_load_queue(self):
        if len(self.load_queue) > 0:
            file = self.load_queue[0]
            self.load_queue = self.load_queue[1:]
            self.list.insert(tk.END,file.name)
            self.fileList.append(file)
            self.empty_load_queue()

