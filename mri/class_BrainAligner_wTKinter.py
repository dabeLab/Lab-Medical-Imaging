from tkinter import *
from tkinter import ttk
import pandas as pd


class BrainAlignerGUI:

    def __init__(self, root):

        root.title("Brain Register")

        # Create outer PanedWindow oriented vertically
        outer_paned_window = ttk.Panedwindow(root, orient=HORIZONTAL)
        outer_paned_window.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Create inner PanedWindows
        inner_paned_window1 = ttk.PanedWindow(outer_paned_window, orient=VERTICAL)
        inner_paned_window2 = ttk.PanedWindow(outer_paned_window, orient=VERTICAL)

        # Add inner PanedWindows to outer PanedWindow
        outer_paned_window.add(inner_paned_window1, weight=1)
        outer_paned_window.add(inner_paned_window2, weight=1)

        # Create frames
        frame1 = ttk.Frame(inner_paned_window1)
        frame1.config(width=500, height=500, relief=RIDGE)  # Images
        frame2 = ttk.Frame(inner_paned_window2)
        frame2.config(width=200, height=250, relief=RIDGE)  # Buttons
        frame3 = ttk.Frame(inner_paned_window2)
        frame3.config(width=200, height=250, relief=RIDGE)  # Log

        # Add frames to inner PanedWindow
        inner_paned_window1.add(frame1)
        inner_paned_window2.add(frame2)
        inner_paned_window2.add(frame3)

        # Frame1 contains the buttons to control the application execution
        self.button_load = ttk.Button(frame2, text="Load", command=self.load_data())
        self.button_load.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.button_reg_atlat = ttk.Button(frame2, text="Register Atlas", command=self.register_atlas())
        self.button_reg_atlat.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.button_reg_mri = ttk.Button(frame2, text="Register MRIs", command=self.register_mris())
        self.button_reg_mri.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        #ttk. #  to set the radius of the brain mask constraint

        # Create a Treeview widget to display the data
        self.tree = ttk.Treeview(inner_paned_window2)
        self.tree.pack(fill='both', expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(root, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscroll=scrollbar.set)

        # Load and display the Excel file
        self.load_database("/Users/berri/Medical Imaging/mri/database.csv")

    def load_database(self, filename):
        try:
            # Read Excel file into a pandas DataFrame
            df = pd.read_csv(filename, sep=";")

            # Display column names as headings in the Treeview
            self.tree["columns"] = list(df.columns)
            for column in df.columns:
                self.tree.heading(column, text=column)

            # Insert data rows into the Treeview
            for index, row in df.iterrows():
                self.tree.insert("", "end", text=index, values=list(row))

        except Exception as e:
            print("Error loading Excel file:", e)

    def register_atlas(self):
        return None
        #self.label.config(text="Uno")

    def register_mris(self):
        return None
        #self.label.config(text="Eins")

    def load_data(self):
        print("wip")

def main():
    root = Tk()
    app = BrainAlignerGUI(root)
    root.mainloop()


if __name__ == "__main__": main()
