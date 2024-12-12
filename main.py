from GUI.index import Index
import tkinter as tk
import asyncio


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    app = Index(loop)
    loop.run_forever()
    loop.close()