import os

class File:
    def __init__(self, name, path, size):
        self.name = name
        self.path = path
        self.size = size

    def __repr__(self):
        return f"File(name={self.name}, path={self.path}, size={self.size} bytes)"


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
        """
        Calculate the total size of all files in this directory and its subdirectories.
        """
        size = sum(file.size for file in self.files)
        size += sum(subdir.total_size() for subdir in self.subdirectories)
        return size

    def __repr__(self):
        return f"Directory(name={self.name}, path={self.path}, files={len(self.files)}, subdirectories={len(self.subdirectories)})"


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
        print(f"{indent}{directory.name}/ ({total_size} bytes)")

        for file in directory.files:
            print(f"{indent}    {file.name} ({file.size} bytes)")

        for subdir in directory.subdirectories:
            self.print_tree(subdir, level + 1)


# Example Usage
if __name__ == "__main__":
    fs = FileSystemWrapper()

    # Build the tree for a directory
    root_path = "example_dir"  # Change this to your target directory
    try:
        tree = fs.build_tree(root_path)

        # Print the tree structure
        print("\nDirectory Structure with Sizes:")
        fs.print_tree()
    except ValueError as e:
        print(e)
