import tkinter as tk

# Scrollable frame implementation
class VerticalScrolledFrame(tk.Frame):
    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)
        canvas = tk.Canvas(self, borderwidth=0, width=200, height=kw['height']-100)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")