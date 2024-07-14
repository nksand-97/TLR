import tkinter as tk
from tkinter import simpledialog
import customtkinter as ctk
import threading
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk
from ultralytics.utils.plotting import Annotator, colors
from utils.general import scale_boxes

from cap import Capture
from rcg import Recognition


# Const
font_type               = "meiryo"
window_size             = "1000x600"
app_title               = "TLR Viewer"
app_brightness_theme    = "dark"
app_color_theme         = "blue"
update_cycle            = 50
line_thickness          = 2


# Viewer class
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Tkinter application setup
        self.fonts = {
            "L": (font_type, 30),
            "M": (font_type, 20),
            "S": (font_type, 10)
        }
        self.geometry(window_size)
        self.title(app_title)
        ctk.set_appearance_mode(app_brightness_theme)
        ctk.set_default_color_theme(app_color_theme)

        # Initialize variable
        self.update_cycle = update_cycle    # Update cycle
        self.running = False    # Flag to turn on/off the recognition
        self.curr_image = None  # Current webcam image (Numpy array style) 
        self.base_time = datetime.now() # Base for application time
        self.cap = Capture(self)
        self.rcg = Recognition(self)

        # Setup application layout
        self.layoutSetting()

        # Start threading
        self.startThreading()


    # Setup application layout
    def layoutSetting(self):
        # Grid layout configuration
        self.grid_rowconfigure(0, weight=2)
        self.grid_rowconfigure(1, weight=13)
        self.grid_rowconfigure(2, weight=6)
        self.grid_rowconfigure(3, weight=6)
        self.grid_columnconfigure(0, weight=25)
        self.grid_columnconfigure(1, weight=95)
        self.grid_columnconfigure(2, weight=32)

        # Capture label
        self.captureLabel = ctk.CTkLabel(self, text="Capture", font=self.fonts["M"])
        self.captureLabel.grid(row=0, column=0, sticky="w")

        # Info label
        self.infoLabel = ctk.CTkLabel(self, text="Info", font=self.fonts["M"])
        self.infoLabel.grid(row=0, column=2, sticky="w")

        # Capture canvas
        self.captureCanvas = ctk.CTkCanvas(self)
        self.captureCanvas.grid(row=1, column=0, rowspan=2, columnspan=2, sticky="nsew")

        # Info frame
        self.infoFrame = ctk.CTkFrame(self)
        self.infoFrame.grid(row=1, column=2, sticky="nsew")

        self.infoFrame.grid_rowconfigure(0, weight=1)
        self.infoFrame.grid_rowconfigure(1, weight=1)
        self.infoFrame.grid_rowconfigure(2, weight=1)
        self.infoFrame.grid_rowconfigure(3, weight=3)
        self.infoFrame.grid_columnconfigure(0, weight=1)

        # Info 1 : recognition speed
        self.info1 = ctk.CTkLabel(self.infoFrame, text="T=000ms", font=self.fonts["L"])
        self.info1.grid(row=0, column=0, sticky="ew")

        # Info 2 : current time
        self.info2 = ctk.CTkLabel(
            self.infoFrame, text=str(datetime.now().time())[:8], font=self.fonts["L"]
        )
        self.info2.grid(row=1, column=0, sticky="ew")

        # Info 3 : application time
        self.info3 = ctk.CTkLabel(self.infoFrame, text="00:00:00", font=self.fonts["L"])
        self.info3.grid(row=2, column=0, sticky="ew")

        # Info 4 : recognition speed
        self.info4 = ctk.CTkLabel(
            self.infoFrame, text="Stopped.", font=self.fonts["L"], fg_color="red"
        )
        self.info4.grid(row=3, column=0, sticky="nsew")

        # Start button
        self.startButton = ctk.CTkButton(self, text="START", command=self.switchState)
        self.startButton.grid(row=2, column=2, sticky="nsew")

        # Result label
        self.resultLabel = ctk.CTkLabel(self, text="Result", font=self.fonts["M"])
        self.resultLabel.grid(row=3, column=0, sticky="e")

        # Result canvas
        self.resultCanvas = ctk.CTkCanvas(self, bg="red", height=0)
        self.resultCanvas.grid(row=3, column=1, sticky="nsew")

        # Quit button
        self.quitButton = ctk.CTkButton(self, text="QUIT", command=self.quit)
        self.quitButton.grid(row=3, column=2, sticky="nsew")
    

    # Start threading
    def startThreading(self):
        th_main = threading.Thread(target=self.updateGUI)
        th_main.start()

        th_cap = threading.Thread(target=self.cap.update)
        th_cap.daemon = True
        th_cap.start()

        th_rcg = threading.Thread(target=self.rcg.update)
        th_rcg.daemon = True
        th_rcg.start()


    # Update every XXms
    def updateGUI(self):
        # Current time
        self.info2.configure(text=str(datetime.now().time())[:8])

        # Captured frame
        if self.cap.frame is not None:
            self.curr_image = self.cap.frame.copy()
            frame_disp = cv2.resize(
                self.curr_image,
                (self.captureCanvas.winfo_width(), self.captureCanvas.winfo_height())
            )

        # Recognition ON
        if self.running:
            # Application time
            elapsed_time = datetime.now() - self.base_time
            h, r = divmod(elapsed_time.seconds, 3600)
            m, s = divmod(r, 60)
            self.info3.configure(text=f"{h:02}:{m:02}:{s:02}")
            
            # Draw recognition results
            if (self.cap.frame is not None) and (self.rcg.det is not None):
                det = self.rcg.det
                im0 = self.curr_image.copy()
                annotator = Annotator(
                    im0, line_width=line_thickness, example=str(self.rcg.model.names)
                )
                if len(det):
                    det[:, :4] = scale_boxes(
                        self.rcg.img_tensor.shape[2:],
                        det[:, :4],
                        im0.shape
                    ).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = self.rcg.model.names[c]
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                frame_disp = cv2.resize(
                    im0, (self.captureCanvas.winfo_width(), self.captureCanvas.winfo_height())
                )

        img = Image.fromarray(frame_disp)
        imgtk = ImageTk.PhotoImage(image=img)
        self.imgtk = imgtk
        self.captureCanvas.create_image(0, 0, image=self.imgtk, anchor = tk.NW)

        self.after(self.update_cycle, self.updateGUI)


    # Switch recognition state if Start/Stop button pushed
    def switchState(self):
        # ON -> OFF
        if self.running:
            self.running = False
            self.info3.configure(text="00:00:00")
            self.info4.configure(text="Stopped", fg_color="red")
            self.startButton.configure(text="START")

        # OFF -> ON
        else:
            self.running = True
            self.base_time = datetime.now()
            self.info4.configure(text="Running", fg_color="green")
            self.startButton.configure(text="STOP")


    # Stop this application
    def quit(self):
        if self.cap.cap is not None:
            self.cap.cap.release()
        self.after(100, self.destroy)