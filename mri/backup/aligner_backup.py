import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import SimpleITK as sitk


class BrainAligner:

    def __init__(self, fix_img: sitk.Image, mov_img: sitk.Image):

        self.ax = None
        self.fig = None
        self.ax_mov_img_yz = None
        self.ax_mov_img_xz = None
        self.ax_mov_img_xy = None
        self.ax_fix_img_yz = None
        self.ax_fix_img_xz = None
        self.ax_fix_img_xy = None
        self.fix_img = fix_img
        self.mov_img = mov_img
        self.click1 = None
        self.click2 = None
        self.fix_img_xyz = np.array([fix_img.GetSize()[0] // 2, fix_img.GetSize()[1] // 2, fix_img.GetSize()[2] // 2], dtype=sitk.VectorUInt32)  # current coordinates of fixed image center
        self.mov_img_xyz = np.array([mov_img.GetSize()[0] // 2, mov_img.GetSize()[1] // 2, mov_img.GetSize()[2] // 2], dtype=sitk.VectorUInt32)  # current coordinates of moving image center

    def execute(self):
        """In the order, the function creates a figure with 3 subplots (xy, xz and yz planes). Then, plots sections of the fixed image (MR image)
        and lets the user find the brain center by clicking on the MR sections. For example, the user can find the center on the z-axis by
        clicking on the xz sections. This automatically acquires the x and y coordinates, and set them as the new origin. Then, it updates the plots
        with the sections passing through the new coordinates. The centering continues until the user is satisfied with their selection of
        coordinates and press "Return". Then the functions plots an"""
        self.create_figure()
        self.place_fix_img()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_center)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_return)
        mplcursors.cursor(hover=True)
        plt.tight_layout()
        plt.show(block=False)


        self.fig.canvas.mpl_connect('button_press_event', self.on_click_move)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        mplcursors.cursor(hover=True)
        plt.tight_layout()
        plt.show(block=False)
        self.fig.canvas.start_event_loop()
    
    def create_figure(self):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(1, 3)
        for idx, item in enumerate(self.ax.flatten()):
            item.set_axis_off()
        # label axes - necessary for identifying the axes
        self.ax[0].set_label("xy")
        self.ax[1].set_label("xz")
        self.ax[2].set_label("yz")
    
    def place_fix_img(self,):
        # Display fixed image - xy plane - Extent order: left, right, bottom, top
        img_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], self.fix_img.GetSize()[1], 0], [0, 0, self.fix_img_xyz[2]])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_fix_img_xy = self.ax[0].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='gray', extent=extent)

        # Display fixed image - xz plane
        img_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], 0, self.fix_img.GetSize()[2]], [0, self.fix_img_xyz[1], 0])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_fix_img_xz = self.ax[1].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='gray', extent=extent)

        # Display fixed image - xy plane
        img_slice = sitk.Extract(self.fix_img, [0, self.fix_img.GetSize()[1], self.fix_img.GetSize()[2]], [self.fix_img_xyz[0], 0, 0])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_fix_img_yz = self.ax[2].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='gray', extent=extent)

    def place_mov_img(self,):
        # Display moving image - xy plane - Extent order: left, right, bottom, top
        img_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], self.mov_img.GetSize()[1], 0], [0, 0, self.mov_img_xyz[2]])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_mov_img_xy = self.ax[0].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='jet', extent=extent, alpha=0.3)

        # Display moving image - xz plane
        img_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], 0, self.mov_img.GetSize()[2]], [0, self.mov_img_xyz[1], 0])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_mov_img_xz = self.ax[1].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='jet', extent=extent, alpha=0.3)

        # Display moving image - xy plane
        img_slice = sitk.Extract(self.mov_img, [0, self.mov_img.GetSize()[1], self.mov_img.GetSize()[2]], [self.mov_img_xyz[0], 0, 0])
        extent = [0, (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1], 0]
        self.ax_mov_img_yz = self.ax[2].imshow(sitk.GetArrayViewFromImage(img_slice), cmap='jet', extent=extent, alpha=0.3)

    def on_close(self,):
        self.fig.canvas.stop_event_loop()

    def on_click_move(self, event):
        if self.click1 is None:
            self.click1 = (event.xdata, event.ydata, event.inaxes.get_label())
            print(f'Click 1 at ({self.click1[0]}, {self.click1[1]}) on {self.click1[2]}')
        else:
            self.click2 = (event.xdata, event.ydata, event.inaxes.get_label())
            print(f'Click 2 at ({self.click2[0]}, {self.click2[1]}) on {self.click2[2]}')
            if self.click1[2] != self.click2[2]:
                print("Clicks must be oin the same section.")
                self.on_click_move()
            self.move_image()

    def on_click_center(self, event):
        self.click1 = (event.xdata, event.ydata, event.inaxes.get_label())
        print(f'Click 1 at ({self.click1[0]}, {self.click1[1]}) on {self.click1[2]}')
        self.center_image()

    def on_key_return(self, event):
        if event.key == "enter":
            print(f"Fix image new origin: {self.fix_img_xyz}")
            self.place_fix_img()

    def center_image(self):
        # Calculate the offset: target location - starting location
        if self.click1[2] == "xy":
            dx = int(self.click1[0] - self.fix_img_xyz[0])
            dy = int(self.click1[1] - self.fix_img_xyz[1])
            dz = 0
            # update xyz coordinates
            self.fix_img.SetOrigin((self.click1[0], self.click1[1], self.fix_img.GetOrigin()[2]))
            #self.fix_img_xyz[0] = self.click1[0]
            #self.fix_img_xyz[1] = self.click1[1]
        elif self.click1[2] == "xz":
            dx = int(self.click1[0] - self.fix_img_xyz[0])
            dy = 0
            dz = int(self.click1[1] - self.fix_img_xyz[2])
            # update xyz coordinates
            self.fix_img_xyz[0] = self.click1[0]
            self.fix_img_xyz[2] = self.click1[2]
        elif self.click1[2] == "yz":
            dx = 0
            dy = int(self.click1[1] - self.fix_img_xyz[1])
            dz = int(self.click1[1] - self.fix_img_xyz[2])
            # update xyz coordinates
            self.fix_img_xyz[1] = self.click1[1]
            self.fix_img_xyz[2] = self.click1[2]
        else:
            dx = 0
            dy = 0
            dz = 0
        # Reset the click variables
        print(f"dx, dy, dz: {dx}, {dy}, {dz}")
        self.update_plot(self.fix_img, dx, dy, dz)
        self.click1 = None

    def move_image(self):
        """"""
        # Calculate the offset: target location - starting location
        if self.click1[2] == self.click2[2] == "xy":
            dx = int(self.click2[0] - self.click1[0])
            dy = int(self.click2[1] - self.click1[1])
            dz = 0
            print(f"dx, dy, dz: {dx}, {dy}, {dz}")
            self.update_plot(dx, dy, dz)

        if self.click1[2] == self.click2[2] == "xz":
            dx = int(self.click2[0] - self.click1[0])
            dy = 0
            dz = int(self.click2[1] - self.click1[1])
            print(f"dx, dy, dz: {dx}, {dy}, {dz}")
            self.update_plot(dx, dy, dz)

        if self.click1[2] == self.click2[2] == "yz":
            dx = 0
            dy = int(self.click2[0] - self.click1[0])
            dz = int(self.click2[1] - self.click1[1])
            print(f"dx, dy, dz: {dx}, {dy}, {dz}")
            self.update_plot(dx, dy, dz)

        # Reset the click variables
        self.click1 = None
        self.click2 = None

    def update_plot_backup(self, img: sitk.Image, dx: int, dy: int, dz: int):

        # update xy
        img_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], self.mov_img.GetSize()[1], 0], [0, 0, self.mov_img_xyz[2] + dz])
        extent = [0 - dx, (0 + img_slice.GetSize()[0] - dx) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dy) * img_slice.GetSpacing()[1], 0 - dy]
        self.ax_mov_img_xy.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_xy.set_extent(extent)

        # update xz
        img_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], 0, self.mov_img.GetSize()[2]], [0, self.mov_img_xyz[1] + dy, 0])
        extent = [0 - dx, (0 + img_slice.GetSize()[0] - dx) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dz) * img_slice.GetSpacing()[1], 0 - dz]
        self.ax_mov_img_xz.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_xz.set_extent(extent)

        # update yz
        img_slice = sitk.Extract(self.mov_img, [0, self.mov_img.GetSize()[1], self.mov_img.GetSize()[2]], [self.mov_img_xyz[0] + dx, 0, 0])
        extent = [0 - dy, (0 + img_slice.GetSize()[0] - dy) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dz) * img_slice.GetSpacing()[1], 0 - dz]
        self.ax_mov_img_yz.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_yz.set_extent(extent)

        self.fig.canvas.draw()

    def update_plot(self, img: sitk.Image, dx: int, dy: int, dz: int):

        # update xy
        img_slice = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], 0], [0, 0, img.GetOrigin()[2] + dz])
        extent = [0 - dx, (0 + img_slice.GetSize()[0] - dx) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dy) * img_slice.GetSpacing()[1], 0 - dy]
        self.ax_mov_img_xy.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_xy.set_extent(extent)

        # update xz
        img_slice = sitk.Extract(img, [img.GetSize()[0], 0, img.GetSize()[2]], [0, img.GetOrigin()[1] + dy, 0])
        extent = [0 - dx, (0 + img_slice.GetSize()[0] - dx) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dz) * img_slice.GetSpacing()[1], 0 - dz]
        self.ax_mov_img_xz.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_xz.set_extent(extent)

        # update yz
        img_slice = sitk.Extract(img, [0, img.GetSize()[1], img.GetSize()[2]], [img.GetOrigin() + dx, 0, 0])
        extent = [0 - dy, (0 + img_slice.GetSize()[0] - dy) * img_slice.GetSpacing()[0], (0 - img_slice.GetSize()[1] - dz) * img_slice.GetSpacing()[1], 0 - dz]
        self.ax_mov_img_yz.set_array(sitk.GetArrayFromImage(img_slice))
        self.ax_mov_img_yz.set_extent(extent)

        self.fig.canvas.draw()

