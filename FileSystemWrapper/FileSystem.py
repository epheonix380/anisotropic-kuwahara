import os
from datetime import datetime

BORDER = "=" * 50

class ByteSize:
    def __init__(self, size_in_bytes):
        self.size_in_bytes = size_in_bytes

    def __str__(self):
        return self.format_size()

    def format_size(self):
        units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
        size = self.size_in_bytes
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.2f} {units[unit_index]}"

    def __add__(self, other):
        if isinstance(other, ByteSize):
            return ByteSize(self.size_in_bytes + other.size_in_bytes)
        return ByteSize(self.size_in_bytes + other)

    def __radd__(self, other):
        return self.__add__(other)


class File:
    def __init__(self, name, path, size):
        self.name = name
        self.path = path
        self.size = ByteSize(size)
        self.file_type = os.path.splitext(name)[1] or "Unknown"
        self.created_time = datetime.fromtimestamp(os.path.getctime(path))
        self.modified_time = datetime.fromtimestamp(os.path.getmtime(path))

    def __str__(self):
        return (
            f"{BORDER}\n"
            f"File: {self.name} ({self.size})\n"
            f"  Path: {self.path}\n"
            f"  Type: {self.file_type}\n"
            f"  Created: {self.created_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Modified: {self.modified_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{BORDER}"
        )


class Directory:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.files = []
        self.subdirectories = []

    def add_file(self, file):
        self.files.append(file)

    def add_subdirectory(self, subdir):
        self.subdirectories.append(subdir)

    def total_size(self):
        size = sum(file.size.size_in_bytes for file in self.files)
        size += sum(subdir.total_size().size_in_bytes for subdir in self.subdirectories)
        return ByteSize(size)
    
    def __str__(self):
        total_size = self.total_size()
        num_files = len(self.files)
        num_subdirs = len(self.subdirectories)
        
        return (f"{BORDER}\n"
                f"Directory: {self.name}/ ({total_size})\n"
                f"  Path: {self.path}\n"
                f"  Contains: {num_files} file(s), {num_subdirs} subdirectory(ies)\n"
                f"{BORDER}")

class Tree:
    def __init__(self, root):
        self.root = root

    def __repr__(self):
        return f"Tree(root={self.root})"


class FileSystemWrapper:
    def __init__(self):
        self.tree = None

    def build_tree(self, root_path):
        if not os.path.isdir(root_path):
            raise ValueError(f"Path is not a directory: {root_path}")

        self.tree = Tree(self._build_directory(root_path))
        return self.tree

    def _build_directory(self, path):
        directory = Directory(name=os.path.basename(path), path=path)

        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    file_info = File(
                        name=entry.name,
                        path=entry.path,
                        size=entry.stat().st_size
                    )
                    directory.add_file(file_info)
                elif entry.is_dir():
                    subdir = self._build_directory(entry.path)
                    directory.add_subdirectory(subdir)
        except PermissionError:
            print(f"Permission denied: {path}")
        except OSError as e:
            print(f"Error accessing {path}: {e}")

        return directory

    def print_tree(self, directory=None, level=0):
        if directory is None:
            directory = self.tree.root

        indent = " " * (level * 4)
        total_size = directory.total_size()
        print(f"{indent}└── {directory.name}/ ({total_size})")

        for file in directory.files:
            print(f"{indent}    └── {file.name} ({file.size})")

        for subdir in directory.subdirectories:
            self.print_tree(subdir, level + 1)

    def find_all_with_ext(self, ext, directory=None) -> list[File]:
        """
        Find a file or directory by name.
        """
        if directory is None:
            directory = self.tree.root
        arr = []
        # Check in the current directory
        for file in directory.files:
            if file.name.split(".")[-1] == ext:
                arr.append(file)
        # Recur into subdirectories
        for subdir in directory.subdirectories:
            result = self.find_all_with_ext(ext, subdir)
            if result:
                arr = arr + result

        return arr

    def find(self, name, directory=None):
        """
        Find a file or directory by name.
        """
        if directory is None:
            directory = self.tree.root

        # Check in the current directory
        for file in directory.files:
            if file.name == name:
                return file
        for subdir in directory.subdirectories:
            if subdir.name == name:
                return subdir

        # Recur into subdirectories
        for subdir in directory.subdirectories:
            result = self.find(name, subdir)
            if result:
                return result

        return None

    def get_directory_by_path(self, path):
        """
        Get a directory object by its path.
        """
        parts = path.strip("/").split("/")
        current = self.tree.root

        for part in parts:
            found = None
            for subdir in current.subdirectories:
                if subdir.name == part:
                    found = subdir
                    break
            if found:
                current = found
            else:
                return None  # Directory not found

        return current

    def print_directory(self, path):
        """
        Print a specific directory by its path in a formatted structure.
        """
        directory = self.get_directory_by_path(path)
        if directory:
            self.print_tree(directory)
        else:
            print(f"Directory not found: {path}")


# Example Usage
if __name__ == "__main__":
    fs = FileSystemWrapper()

    # Build the tree for a directory
    root_path = "example_dir"
    try:
        tree = fs.build_tree(root_path)

        # Print the tree structure
        print("\nDirectory Structure with Sizes:")
        fs.print_tree()

        # Find a file or directory
        print("\nFinding 'file_1.txt':")
        result = fs.find("file_1.txt")
        print(result)

        # Get and print a specific directory
        print("\nPrinting 'example_dir/subdir':")
        fs.print_directory("example_dir/subdir")
        
    except ValueError as e:
        print(e)
