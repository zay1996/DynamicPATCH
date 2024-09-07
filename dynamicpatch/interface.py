'''
This script creates the user interface to read in parameters and display outputs
The code is generated with the help of ChatGPT 

@Author: Aiyin Zhang
'''

from dynamicpatch import config
import tkinter as tk
from tkinter import filedialog, messagebox,ttk
import os
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

import matplotlib 
matplotlib.use('Agg')
# Function to process the array and parameters
def array_to_dict(array, years):
    if len(years) != array.shape[0]:
        raise ValueError("The length of the years list must match the first dimension of the array")
    result_dict = {year: array[i] for i, year in enumerate(years)}
    return result_dict

def process_inputs(workpath, years, connectivity, presence, nodata, dataset, FileType,study_area):
    print(f"Workpath: {workpath}")
    print(f"Years: {years}")
    print(f"Connectivity: {connectivity}")
    print(f"Presence: {presence}")
    print(f"No Data Value: {nodata}")
    print(f"File Type: {FileType}")
    print(f"Study Area:{study_area}")
    
    

# GUI setup
class InputApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Parameters")

        # Initialize attributes
        self.file_type = None
        self.file_path = None
        self.dataset = None
        self.params = {}
        self.loaded_from_file = False  # Initialize to track if parameters were loaded from a file

        # GUI components setup
        self.create_widgets()

    def create_widgets(self):
        
        self.load_button = tk.Button(self, text="Load Parameters", command=self.load_params)
        self.load_button.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        
        self.workpath_label = tk.Label(self, text="Workpath:")
        self.workpath_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.workpath_entry = tk.Entry(self, width=50)
        self.workpath_entry.grid(row=1, column=1, padx=10, pady=10)

        self.folder_button = tk.Button(self, text="Browse Folder", command=self.browse_folder)
        self.folder_button.grid(row=1, column=2, padx=5, pady=10)

        self.file_button = tk.Button(self, text="Browse File", command=self.browse_file)
        self.file_button.grid(row=1, column=3, padx=5, pady=10)

        self.years_label = tk.Label(self, text="Years (comma-separated):")
        self.years_label.grid(row=2, column=0, padx=10, pady=10)
        
        self.years_entry = tk.Entry(self)
        self.years_entry.grid(row=2, column=1, padx=10, pady=10)

        self.connectivity_label = tk.Label(self, text="Connectivity (4 or 8):")
        self.connectivity_label.grid(row=3, column=0, padx=10, pady=10)
        
        self.connectivity_entry = tk.Entry(self)
        self.connectivity_entry.insert(0, "8")  # Default value for Connectivity
        self.connectivity_entry.grid(row=3, column=1, padx=10, pady=10)

        self.presence_label = tk.Label(self, text="Presence (integer):")
        self.presence_label.grid(row=4, column=0, padx=10, pady=10)
        
        self.presence_entry = tk.Entry(self)
        self.presence_entry.insert(0, "1")  # Default value for Presence
        self.presence_entry.grid(row=4, column=1, padx=10, pady=10)

        self.nodata_label = tk.Label(self, text="No Data Value (integer):")
        self.nodata_label.grid(row=5, column=0, padx=10, pady=10)
        
        self.nodata_entry = tk.Entry(self)
        self.nodata_entry.grid(row=5, column=1, padx=10, pady=10)
        
        self.study_area_label = tk.Label(self, text="Study Area:")
        self.study_area_label.grid(row=6, column=0, padx=10, pady=10)

        self.study_area_entry = tk.Entry(self)
        self.study_area_entry.grid(row=6, column=1, padx=10, pady=10)

        self.save_button = tk.Button(self, text="Save Parameters", command=self.save_params)
        self.save_button.grid(row=7, column=1, padx=10, pady=10)

        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.grid(row=8, column=0, columnspan=4, padx=10, pady=10)





    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.workpath_entry.delete(0, tk.END)
            self.workpath_entry.insert(0, folder_path)
            

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
        if self.file_path:
            self.workpath_entry.delete(0, tk.END)
            self.workpath_entry.insert(0, self.file_path)
            
            self.check_file()
            

    def check_file(self):
        # Determine the file type based on the extension
        if os.path.isdir(self.file_path):
            self.file_type = "Folder"
        else:
            _, ext = os.path.splitext(self.file_path)
            ext = ext.lower()
            if ext == ".tif":
                self.file_type = "Tif"
            elif ext == ".xlsx":
                self.file_type = "Excel"
            elif ext == ".csv":
                self.file_type = "Csv"
            else:
                messagebox.showerror("Input Error", "The input file must be of type .tif, .xlsx, or .csv.")
                self.file_type = None
                
            # Extract name of the dataset
            self.dataset = os.path.basename(self.file_path).split('.')[0]
            self.study_area_entry.delete(0, tk.END)
            self.study_area_entry.insert(0, self.dataset)
            

    def validate_integer(self, entry):
        try:
            value = int(entry.get())
            return True, value
        except ValueError:
            return False, None

    def validate_connectivity(self, entry):
        value = entry.get()
        if value in ['4', '8']:
            return True, int(value)
        else:
            return False, None

    def update_params(self):
        workpath = self.workpath_entry.get()
        years = self.years_entry.get().split(',')
        self.years = years
        try:
            years = [int(year) for year in years]
        except ValueError:
            messagebox.showerror("Input Error", "Years must be a comma-separated list of integers.")
            return
        
        valid_conn, connectivity = self.validate_connectivity(self.connectivity_entry)
        if not valid_conn:
            messagebox.showerror("Input Error", "Connectivity must be either 4 or 8.")
            return

        valid_presence, presence = self.validate_integer(self.presence_entry)
        if not valid_presence:
            messagebox.showerror("Input Error", "Presence must be an integer.")
            return

        valid_nodata, nodata = self.validate_integer(self.nodata_entry)
        if not valid_nodata:
            messagebox.showerror("Input Error", "No Data Value must be an integer.")
            return        

        if not self.loaded_from_file:
            workpath = self.workpath.get()
            
            # Check if the workpath is a directory (i.e., a folder)
            if os.path.isdir(workpath):
                self.file_type = "Folder"
            
            # Validate the file type if it's not a folder
            if self.file_type != "Folder" and (not hasattr(self, 'file_type') or self.file_type is None):
                messagebox.showerror("Input Error", "Please select a valid file with .tif, .xlsx, or .csv extension.")
                return
        
        study_area = self.study_area_entry.get()
        
        self.params = {
            'workpath': workpath,
            'years': years,
            'connectivity': connectivity,
            'presence': presence,
            'nodata': nodata,
            'FileType': self.file_type,
            'dataset': self.dataset, 
            'study_area':study_area
        }
        
    def submit(self):

        self.update_params()

        # Close the GUI window cleanly
        self.quit()
        self.destroy()  # Close the current window

    def save_params(self):
        self.update_params()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(self.params, file, indent=4)
            messagebox.showinfo("Success", "Parameters saved successfully.")

    def load_params(self):
        params_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if params_path:
            with open(params_path, 'r') as file:
                self.params = json.load(file)
            self.populate_entries()
            self.loaded_from_file = True
        self.file_path = self.params['workpath']
        self.check_file()
            
            

    def populate_entries(self):
        self.workpath_entry.delete(0, tk.END)
        self.workpath_entry.insert(0, self.params['workpath'])

        self.years_entry.delete(0, tk.END)
        self.years_entry.insert(0, ','.join(map(str, self.params['years'])))

        self.connectivity_entry.delete(0, tk.END)
        self.connectivity_entry.insert(0, str(self.params['connectivity']))

        self.presence_entry.delete(0, tk.END)
        self.presence_entry.insert(0, str(self.params['presence']))

        self.nodata_entry.delete(0, tk.END)
        self.nodata_entry.insert(0, str(self.params['nodata']))

        self.study_area_entry.delete(0, tk.END)
        self.study_area_entry.insert(0, str(self.params['study_area']))


    def get_params(self):
        return self.params


class MapApp(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self. setup_ui()
        
    def setup_ui(self):
        self.pattern = None
        self.data_ = None
        self.title("Map and Chart Viewer")
        self.geometry("1200x600")
        self.years = config.year
        self.init_map_index = 0
        self.result_map_index = 0
        self.current_chart_index = 0
        self.map_title_label = None
        
        self.init_map_figs = []
        self.result_map_figs = []
        self.map_title = None
        self.chart_figs = []
        self.chart_titles = []

        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Create Frames for each tab
        self.dynapatch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dynapatch_frame, text='DynaPATCH')

        # Initialize map display
        self.init_canvas = None
        self.result_map_canvas = None
        self.chart_canvas = None
        #self.initial_maps()


        
        # Run Analysis button
        self.run_analysis_button = ttk.Button(self.dynapatch_frame, text="Run Analysis", command=self.start_analysis)
        self.run_analysis_button.pack(pady=20)

        # Progress bar
        self.progress = ttk.Progressbar(self.dynapatch_frame, mode='determinate')
        self.progress.pack(pady=10)

        # Options for analysis
        self.show_transition_var = tk.BooleanVar(value=True)
        self.show_charts_var = tk.BooleanVar(value=True)

        self.option_frame = ttk.Frame(self.dynapatch_frame)
        self.option_frame.pack(pady=20)

        self.show_transition_check = ttk.Checkbutton(
            self.option_frame, text="Show Transition Pattern Maps", variable=self.show_transition_var
        )
        self.show_transition_check.grid(row=0, column=0)

        self.show_charts_check = ttk.Checkbutton(
            self.option_frame, text="Show Charts", variable=self.show_charts_var
        )
        self.show_charts_check.grid(row=0, column=1)
        self.show_map_buttons('init')
        self.initial_maps()
        
                
        #self.protocol("WM_DELETE_WINDOW", self.on_closing)

            
    def initial_maps(self):
        from dynamicpatch.processing import initialize
        self.init_map_figs = initialize()
        #self.show_map()

        self.show_map('init')

    def show_map_buttons(self,flag):
        
        if (flag == 'init'):
            frame = self.dynapatch_frame
            index = self.init_map_index
        elif(flag == 'result'):
            frame = self.result_map_frame
            index = self.result_map_index

        self.next_button = ttk.Button(frame, text="Next >>", command=lambda: self.show_next_map(flag))
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.prev_button = ttk.Button(frame, text="<< Previous", command=lambda: self.show_prev_map(flag))
        self.prev_button.pack(side=tk.LEFT, padx=10)
        
    def show_export_button(self,frame):
        self.export_button = ttk.Button(frame, text = "Export to GeoTIFF",command = self.export_to_geotiff)
        self.export_button.pack(pady = 10)
        
    def export_to_geotiff(self):
        #import WriteData
        file_path = filedialog.asksaveasfilename(defaultextension='.tif', filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")])
        if file_path:
            try:
                from dynamicpatch.processing import write_image
                write_image(self.pattern,file_path)
                messagebox.showinfo("Success", "Map exported successfully as GeoTIFF.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export map: {e}")        
            
    def show_chart_buttons(self,frame):
        # Add previous and next buttons for charts
        if not hasattr(self, 'prev_button_chart') or not hasattr(self, 'next_button_chart'):
            self.prev_button_chart = ttk.Button(frame, text="<< Previous", command=self.show_prev_chart)
            self.prev_button_chart.pack(side=tk.LEFT, padx=10)
    
            self.next_button_chart = ttk.Button(frame, text="Next >>", command=self.show_next_chart)
            self.next_button_chart.pack(side=tk.RIGHT, padx=10)
            
    def show_map(self,flag):
        if (flag == 'init'):
            map_figs = self.init_map_figs
            canvas = self.init_canvas
            frame = self.dynapatch_frame
            index = self.init_map_index
        elif(flag == 'result'):
            map_figs = self.result_map_figs
            canvas = self.result_map_canvas
            frame = self.result_map_frame
            index = self.result_map_index
        #print(index)
        if canvas:
            canvas.get_tk_widget().destroy()
            
        if map_figs:
            #print("run")
            fig = map_figs[index]            
            #print(fig)
            # Create and display the map canvas
            if(flag == 'init'):
                self.init_canvas = FigureCanvasTkAgg(fig, master=frame)
                self.init_canvas.draw()
                self.init_canvas.get_tk_widget().pack(fill='both', expand=True)
                canvas = self.init_canvas
            #print(f"canvas after update:{canvas}")
            if(flag == 'result'):
                self.result_map_canvas = FigureCanvasTkAgg(fig, master=frame)
                self.result_map_canvas.draw()
                self.result_map_canvas.get_tk_widget().pack(fill='both', expand=True)     
                canvas = self.result_map_canvas
            # Bind the save event to the map canvas
            
    
            self.bind_save_event(canvas)
            
    def show_chart(self,frame):
        # Remove the previous chart and title
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().pack_forget()
    
        if hasattr(self, 'chart_title_label'):
            self.chart_title_label.pack_forget()
    
        if self.chart_figs:
            fig = self.chart_figs[self.current_chart_index]  # Fetch the chart for the current index
            
            # Add title display
            title = self.chart_titles[self.current_chart_index]  # Fetch the title for the current index
            self.chart_title_label = ttk.Label(frame, text=title, font=("Arial", 14))
            self.chart_title_label.pack(pady=10)  # Place the title at the top with some padding
            
            # Display the chart
            self.chart_canvas = FigureCanvasTkAgg(fig, master=frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack()
            self.bind_save_event(self.chart_canvas) 
            
            
    def show_prev_map(self,flag):
        if (flag == 'init'):
            index = self.init_map_index
        elif(flag == 'result'):
            index = self.result_map_index
        if index > 0:
            if(flag == 'init'):
                self.init_map_index -= 1
            if(flag == 'result'):
                self.result_map_index -= 1
            #index -= 1
            self.show_map(flag)

    def show_next_map(self,flag):
        if (flag == 'init'):
            map_figs = self.init_map_figs
            index = self.init_map_index
        elif(flag == 'result'):
            map_figs = self.result_map_figs

            index = self.result_map_index        
        #print(flag,len(map_figs),index)
        if index < len(map_figs) - 1:
            if(flag == 'init'):
                self.init_map_index += 1
            if(flag == 'result'):
                self.result_map_index += 1
            #index += 1
            #print(index)
            self.show_map(flag)


    def show_prev_chart(self):
        if self.current_chart_index > 0:
            self.current_chart_index -= 1
            self.show_chart(self.notebook.nametowidget(self.notebook.select()))

    def show_next_chart(self):
        if self.current_chart_index < len(self.chart_figs) - 1:
            self.current_chart_index += 1
            self.show_chart(self.notebook.nametowidget(self.notebook.select()))

    def bind_save_event(self, canvas):
        def save_figure(event):
            file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                canvas.figure.savefig(file_path)

        canvas.get_tk_widget().bind("<Button-3>", save_figure)

    '''
    def run_analysis_thread(self):


        self.progress.start()  # Start the progress bar
        analysis_thread = threading.Thread(target = self.run_analysis_ui)
        analysis_thread.start()
        if analysis_thread.is_alive():
            self.update()  # This keeps the UI responsive
            self.after(50)  # Pause for 50 milliseconds
        else:
            self.progress.stop()  # Stop the progress bar after the thread finishes
            
    def run_analysis_ui(self):
        from processing import run_analysis
        mapshow = self.show_transition_var.get()
        chartsshow = self.show_charts_var.get()

        if mapshow or chartsshow:
            self.pattern, self.map_figs, self.map_title, self.chart_figs,\
                self.chart_titles = run_analysis(mapshow=mapshow, chartsshow=chartsshow)
            self.current_map_index = 0 
            self.current_chart_index = 0
            self.after(0, self.update_ui_after_analysis)
    '''
    def start_analysis(self):
        self.progress.start()
        self.run_analysis_button.config(state='disabled')
        self.update()
        #self.initial_map_figs = self.init_map_figs.copy()
        self.after(100, self.run_analysis_step)
        
    def run_analysis_step(self):
        from dynamicpatch.processing import run_analysis
        try:
            mapshow = self.show_transition_var.get()
            chartsshow = self.show_charts_var.get()
            print(mapshow,chartsshow)
            if not mapshow and not chartsshow:
                messagebox.showerror("Error", "Please select at least one option (Show Transition Pattern Maps or Show Charts)")
                self.progress.stop()
                self.run_analysis_button.config(state='normal')
                return           
        
            # Run a small part of the analysis
            #is_complete, partial_result = run_analysis(progress = self.progress, mapshow=mapshow, chartsshow=chartsshow)
            #if is_complete:
            self.pattern, self.result_map_figs, self.map_title, self.chart_figs, self.chart_titles = run_analysis(progress = self.progress, mapshow=mapshow, chartsshow=chartsshow)
                #self.current_chart_index = 0
            self.after(100, self.update_ui_after_analysis)
            #else:
            #    self.after(100, self.run_analysis_step)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.progress.stop()
            self.run_analysis_button.config(state='normal')     

    def update_ui_after_analysis(self):
        self.progress.stop()
        self.run_analysis_button.config(state='normal')
        messagebox.showinfo("Analysis Complete", "Analysis finished.")

        mapshow = self.show_transition_var.get()
        chartsshow = self.show_charts_var.get()
        
        if mapshow:
            # Create Result - Maps tab 
            self.result_map_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.result_map_frame, text='Result - Maps')
    
    
            '''
            # Create a frame for the map and title
            map_frame = ttk.Frame(self.result_map_frame)
            map_frame.pack(side=tk.TOP, fill='both', expand=True)
            
            # Create a frame for the buttons
            button_frame = ttk.Frame(self.result_map_frame)
            button_frame.pack(side=tk.BOTTOM, pady=5, fill='x')
            '''
    
            # show map on the map frame
            
            self.show_map_buttons('result')
            title = self.map_title
            # Only create the title label if it doesn't already exist
            if hasattr(self, 'map_title_label') and self.map_title_label:
                self.map_title_label.config(text=title)
            
            else:
                self.map_title_label = ttk.Label(self.result_map_frame, text=title, font=("Arial", 14))
                self.map_title_label.pack(side=tk.TOP, pady=10)  # Title at the top with padding
                
            self.show_export_button(self.result_map_frame)
            self.show_map('result')
        
        
        if chartsshow:
            # Create the chart tab and display charts
            self.result_chart_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.result_chart_frame, text='Result - Charts')
            
            # Display chart-related buttons and the chart itself
            self.show_chart_buttons(self.result_chart_frame)
            self.show_chart(self.result_chart_frame)
        
        if mapshow:
            # Automatically switch to the Result - Maps tab
            self.notebook.select(self.result_map_frame)
        elif chartsshow:
            self.notebook.select(self.result_chartsframe)
        # Reset current indices
        #self.result_map_index = 0
        self.current_chart_index = 0    
    def on_closing(self):
        self.destroy()
        self.quit()  # Ensure the program exits completely


def main():
    input_app = InputApp()
    input_app.mainloop()
    params = input_app.get_params()
    
    if params:
        root = tk.Tk()
        root.withdraw()
        map_app = MapApp(root)
        map.app.protocol("WM_DELETE_WINDOW",root.quit)
        root.mainloop()


if __name__ == "__main__":
    main()