import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import SimpleITK as sitk
from utilities import custom_colormap


class BrainAligner:

    def __init__(self, fix_img: sitk.Image, mov_img: sitk.Image):
        """
        The physical space is described by the (x, y, z) coordinate system
        The fixed image space is described by the (i, j, k) coordinate system
        The moving image space is described by the (l, m, n) coordinate system
        """

        self.fig1, self.ax1 = self.create_figure1()
        self.fig2, self.ax2 = self.create_figure2()

        self.fix_img = fix_img
        self.mov_img = mov_img

        self.click1 = None
        self.click2 = None

        # Fixed image slice, initial (0) and current
        self.i0, self.j0, self.k0 = fix_img.GetSize()[0] // 2, fix_img.GetSize()[1] // 2, fix_img.GetSize()[2] // 2
        self.i, self.j, self.k = self.i0, self.j0, self.k0
        # Moving image slice, initial (0), and current
        self.l0, self.m0, self.n0 = mov_img.GetSize()[0] // 2, mov_img.GetSize()[1] // 2, mov_img.GetSize()[2] // 2
        self.l, self.m, self.n = self.l0, self.m0, self.n0
        # dx, dy, dz in physical space
        self.delta = 0

        # Transformation
        self.transform1 = None
        self.transform2 = None
        self.transform3 = None
        self.transform = None

    def execute(self):
        """In the order, the function creates a figure with 3 subplots (xy, xz and yz planes). Then, plots sections of the fixed image (MR image)
        and lets the user find the brain center by clicking on the MR sections. For example, the user can find the center on the z-axis by
        clicking on the xz sections. This automatically acquires the x and y coordinates, and set them as the new origin. Then, it updates the plots
        with the sections passing through the new coordinates. The centering continues until the user is satisfied with their selection of
        coordinates and press "Return". Then the functions plots an"""

        self.plot_slices(self.i0, self.j0, self.k0, self.l0, self.m0, self.n0)
        plt.tight_layout()
        self.fig1.canvas.mpl_connect("button_press_event", self.on_click1)
        self.fig2.canvas.mpl_connect("button_press_event", self.on_click2)
        self.fig1.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig2.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show(block=False)
        mplcursors.cursor(hover=True)
        plt.show()

    @staticmethod
    def create_figure1():
        """Create figure and axes for brain centering"""
        fig, ax = plt.subplots(2, 3)
        for idx, item in enumerate(ax.flatten()):
            item.set_axis_off()
        # label axes - necessary for identifying the axes
        ax[0, 0].set_label("xy fix")
        ax[0, 1].set_label("xz fix")
        ax[0, 2].set_label("yz fix")
        ax[1, 0].set_label("xy mov")
        ax[1, 1].set_label("xz mov")
        ax[1, 2].set_label("yz mov")
        fig.suptitle("Click to center the image. Press 'Return' to store the coordinates")
        ax[0, 0].set_title("x-y plane")
        ax[0, 1].set_title("x-z plane")
        ax[0, 2].set_title("y-z plane")
        return fig, ax

    @staticmethod
    def create_figure2():
        """Create figure and axes for brain alignment"""
        fig, ax = plt.subplots(1, 3)
        for idx, item in enumerate(ax.flatten()):
            item.set_axis_off()
        # label axes - necessary for identifying the axes
        ax[0].set_label("xy")
        ax[1].set_label("xz")
        ax[2].set_label("yz")
        fig.suptitle("Click to center the image. Press 'Return' to store the coordinates")
        ax[0].set_title("x-y plane")
        ax[1].set_title("x-z plane")
        ax[2].set_title("y-z plane")
        return fig, ax

    def plot_slices(self, i, j, k, l, m, n):
        """This function cuts slices of fixed and moving images passing through (i, j, k) and (l, m, n), respectively.
        Then, it plots the fixed and moving image slices on separate rows in figure 1, for brain(s) centering, and
        the fixed and moving image overlaid in Figure 2 for brain alignment."""
        # -----------------------------------------------------------------------
        # Display fixed image - xy plane - Extent order: left, right, bottom, top
        fix_img_xy_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], self.fix_img.GetSize()[1], 0], [0, 0, k])
        extent = [0,
                  (0 + fix_img_xy_slice.GetSize()[0]) * fix_img_xy_slice.GetSpacing()[0],
                  (0 - fix_img_xy_slice.GetSize()[1]) * fix_img_xy_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 0].imshow(sitk.GetArrayViewFromImage(fix_img_xy_slice), cmap='gray', extent=extent)

        # Display fixed image - xz plane
        fix_img_xz_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], 0, self.fix_img.GetSize()[2]], [0, j, 0])
        extent = [0,
                  (0 + fix_img_xz_slice.GetSize()[0]) * fix_img_xz_slice.GetSpacing()[0],
                  (0 - fix_img_xz_slice.GetSize()[1]) * fix_img_xz_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 1].imshow(sitk.GetArrayViewFromImage(fix_img_xz_slice), cmap='gray', extent=extent)

        # Display fixed image - xy plane
        fix_img_yz_slice = sitk.Extract(self.fix_img, [0, self.fix_img.GetSize()[1], self.fix_img.GetSize()[2]], [i, 0, 0])
        extent = [0,
                  (0 + fix_img_yz_slice.GetSize()[0]) * fix_img_yz_slice.GetSpacing()[0],
                  (0 - fix_img_yz_slice.GetSize()[1]) * fix_img_yz_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 2].imshow(sitk.GetArrayViewFromImage(fix_img_yz_slice), cmap='gray', extent=extent)

        # -----------------------------------------------------------------------
        # Display moving image - xy plane - Extent order: left, right, bottom, top
        mov_img_xy_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], self.mov_img.GetSize()[1], 0], [0, 0, n])
        extent = [0,
                  (0 + mov_img_xy_slice.GetSize()[0]) * mov_img_xy_slice.GetSpacing()[0],
                  (0 - mov_img_xy_slice.GetSize()[1]) * mov_img_xy_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 0].imshow(sitk.GetArrayViewFromImage(mov_img_xy_slice), cmap='gray', extent=extent)

        # Display fixed image - xz plane
        mov_img_xz_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], 0, self.mov_img.GetSize()[2]], [0, m, 0])
        extent = [0,
                  (0 + mov_img_xz_slice.GetSize()[0]) * mov_img_xz_slice.GetSpacing()[0],
                  (0 - mov_img_xz_slice.GetSize()[1]) * mov_img_xz_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 1].imshow(sitk.GetArrayViewFromImage(mov_img_xz_slice), cmap='gray', extent=extent)

        # Display fixed image - xy plane
        mov_img_yz_slice = sitk.Extract(self.mov_img, [0, self.mov_img.GetSize()[1], self.mov_img.GetSize()[2]], [l, 0, 0])
        extent = [0,
                  (0 + mov_img_yz_slice.GetSize()[0]) * mov_img_yz_slice.GetSpacing()[0],
                  (0 - mov_img_yz_slice.GetSize()[1]) * mov_img_yz_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 2].imshow(sitk.GetArrayViewFromImage(mov_img_yz_slice), cmap='gray', extent=extent)

        # -----------------------------------------------------------------------
        # Display fixed and moving image overlaid in Figure 2.
        self.ax2[0].imshow(sitk.GetArrayViewFromImage(fix_img_xy_slice), cmap='gray', extent=extent)
        self.ax2[0].imshow(sitk.GetArrayViewFromImage(mov_img_xy_slice), cmap=custom_colormap(), extent=extent)
        self.ax2[1].imshow(sitk.GetArrayViewFromImage(fix_img_xz_slice), cmap='gray', extent=extent)
        self.ax2[1].imshow(sitk.GetArrayViewFromImage(mov_img_xz_slice), cmap=custom_colormap(), extent=extent)
        self.ax2[2].imshow(sitk.GetArrayViewFromImage(fix_img_yz_slice), cmap='gray', extent=extent)
        self.ax2[2].imshow(sitk.GetArrayViewFromImage(mov_img_yz_slice), cmap=custom_colormap(), extent=extent)

    def update_plot(self):
        """Update the displayed slices."""
        # -----------------------------------------------------------------------
        # Update fixed image - xy plane
        fix_img_xy_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], self.fix_img.GetSize()[1], 0], [0, 0, self.k])
        extent = [0,
                  (0 + fix_img_xy_slice.GetSize()[0]) * fix_img_xy_slice.GetSpacing()[0],
                  (0 - fix_img_xy_slice.GetSize()[1]) * fix_img_xy_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 0].images[0].set_data(sitk.GetArrayFromImage(fix_img_xy_slice))
        self.ax1[0, 0].images[0].set_extent(extent)

        # Update fixed image - xz plane
        fix_img_xz_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], 0, self.fix_img.GetSize()[2]], [0, self.j, 0])
        extent = [0,
                  (0 + fix_img_xz_slice.GetSize()[0]) * fix_img_xz_slice.GetSpacing()[0],
                  (0 - fix_img_xz_slice.GetSize()[1]) * fix_img_xz_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 1].images[0].set_data(sitk.GetArrayFromImage(fix_img_xz_slice))
        self.ax1[0, 1].images[0].set_extent(extent)

        # Update fixed image - yz plane
        fix_img_yz_slice = sitk.Extract(self.fix_img, [0, self.fix_img.GetSize()[1], self.fix_img.GetSize()[2]], [self.i, 0, 0])
        extent = [0,
                  (0 + fix_img_yz_slice.GetSize()[0]) * fix_img_yz_slice.GetSpacing()[0],
                  (0 - fix_img_yz_slice.GetSize()[1]) * fix_img_yz_slice.GetSpacing()[1],
                  0]
        self.ax1[0, 2].images[0].set_data(sitk.GetArrayFromImage(fix_img_yz_slice))
        self.ax1[0, 2].images[0].set_extent(extent)

        # -----------------------------------------------------------------------
        # Update moving image - xy plane
        mov_img_xy_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], self.mov_img.GetSize()[1], 0], [0, 0, self.n])
        extent = [0,
                  (0 + mov_img_xy_slice.GetSize()[0]) * mov_img_xy_slice.GetSpacing()[0],
                  (0 - mov_img_xy_slice.GetSize()[1]) * mov_img_xy_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 0].images[0].set_data(sitk.GetArrayFromImage(mov_img_xy_slice))
        self.ax1[1, 0].images[0].set_extent(extent)

        # Update moving image - xz plane
        mov_img_xz_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], 0, self.mov_img.GetSize()[2]], [0, self.m, 0])
        extent = [0,
                  (0 + mov_img_xz_slice.GetSize()[0]) * mov_img_xz_slice.GetSpacing()[0],
                  (0 - mov_img_xz_slice.GetSize()[1]) * mov_img_xz_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 1].images[0].set_data(sitk.GetArrayFromImage(mov_img_xz_slice))
        self.ax1[1, 1].images[0].set_extent(extent)

        # Update moving image - yz plane
        mov_img_yz_slice = sitk.Extract(self.mov_img, [0, self.fix_img.GetSize()[1], self.mov_img.GetSize()[2]], [self.l, 0, 0])
        extent = [0,
                  (0 + mov_img_yz_slice.GetSize()[0]) * mov_img_yz_slice.GetSpacing()[0],
                  (0 - mov_img_yz_slice.GetSize()[1]) * mov_img_yz_slice.GetSpacing()[1],
                  0]
        self.ax1[1, 2].images[0].set_data(sitk.GetArrayFromImage(mov_img_yz_slice))
        self.ax1[1, 2].images[0].set_extent(extent)

        # -----------------------------------------------------------------------
        # Display fixed and moving image overlaid in Figure 2.
        self.ax2[0].imshow(sitk.GetArrayViewFromImage(fix_img_xy_slice), cmap='gray', extent=extent)
        self.ax2[0].imshow(sitk.GetArrayViewFromImage(mov_img_xy_slice), cmap=custom_colormap(), extent=extent)
        self.ax2[1].imshow(sitk.GetArrayViewFromImage(fix_img_xz_slice), cmap='gray', extent=extent)
        self.ax2[1].imshow(sitk.GetArrayViewFromImage(mov_img_xz_slice), cmap=custom_colormap(), extent=extent)
        self.ax2[2].imshow(sitk.GetArrayViewFromImage(fix_img_yz_slice), cmap='gray', extent=extent)
        self.ax2[2].imshow(sitk.GetArrayViewFromImage(mov_img_yz_slice), cmap=custom_colormap(), extent=extent)

        self.fig1.canvas.draw()
        self.fig2.canvas.draw()

    def on_click1(self, event):
        """It requires one click on any axis in the 'Brain Centering' figure. Then, it calculates and plots the 
        three slices passing through the selected pair of coordinates, and calculate the transformation required to 
        translate either the fixed or moving image so to have the brain center in the center."""""

        self.click1 = (event.xdata, event.ydata, event.inaxes.get_label(), event.inaxes.figure.number)
        print(f'Click 1 at ({self.click1[0]}, {self.click1[1]}) on {self.click1[2]}, fig {self.click1[3]}')

        if self.click1[2] == "xy fix":
            self.i = int(self.click1[0] // self.fix_img.GetSpacing()[0])
            self.j = int(-1 * self.click1[1] // self.fix_img.GetSpacing()[1])
        if self.click1[2] == "xz fix":
            self.i = int(self.click1[0] // self.fix_img.GetSpacing()[0])
            self.k = int(-1 * self.click1[1] // self.fix_img.GetSpacing()[2])
        if self.click1[2] == "yz fix":
            self.j = int(self.click1[0] // self.fix_img.GetSpacing()[1])
            self.k = int(-1 * self.click1[1] // self.fix_img.GetSpacing()[2])
        if self.click1[2] == "xy mov":
            self.l = int(self.click1[0] // self.mov_img.GetSpacing()[0])
            self.m = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[1])
        if self.click1[2] == "xz mov":
            self.l = int(self.click1[0] // self.mov_img.GetSpacing()[0])
            self.n = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[2])
        if self.click1[2] == "yz mov":
            self.m = int(self.click1[0] // self.mov_img.GetSpacing()[1])
            self.n = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[2])

        #self.update_plot()
        self.click1 = None

        # Get coordinates in physical space
        xyz1 = np.array(self.fix_img.TransformIndexToPhysicalPoint([self.i, self.j, self.k]))
        xyz2 = np.array(self.mov_img.TransformIndexToPhysicalPoint([self.l, self.m, self.n]))

        # calculate transformation and update attributes
        transform = sitk.TranslationTransform(3)
        transform.SetOffset(xyz1 - xyz2)
        self.update_transform(transform)
        self.mov_img = sitk.Resample(self.mov_img,
                                     transform=self.transform,
                                     interpolator=sitk.sitkLinear,
                                     defaultPixelValue=0,
                                     outputPixelType=self.mov_img.GetPixelIDValue())
        self.update_plot()

    def on_click2(self, event):
        """It requires two clicks on the same axis in the 'Brain Aligner' figure. Then, it calculates the
        transformation required to translate rigidly the moving image with respect to the fixed image."""

        # make a copy of attributes to initialize the function
        i1, j1, k1 = int(self.l), int(self.m), int(self.n)
        i2, j2, k2 = int(self.l), int(self.m), int(self.n)

        if event.inaxes:
            if self.click1 is None:
                self.click1 = (event.xdata, event.ydata, event.inaxes.get_label(), event.inaxes.figure.number)
                print(f'Click 1 at ({self.click1[0]}, {self.click1[1]}) on {self.click1[2]}, fig {self.click1[3]}')

            elif self.click1 is not None and self.click2 is None:
                self.click2 = (event.xdata, event.ydata, event.inaxes.get_label(), event.inaxes.figure.number)
                print(f'Click 2 at ({self.click2[0]}, {self.click2[1]}) on {self.click2[2]} on fig {self.click2[3]}')

                # if the clicks are on the same axis, execute alignment
                if self.click1[3] == self.click2[3] and self.click1[2] == self.click2[2]:

                    # Calculate the offset: target location - starting location
                    if self.click1[2] == self.click2[2] == "xy":
                        i1 = int(self.click1[0] // self.mov_img.GetSpacing()[0])
                        i2 = int(self.click2[0] // self.mov_img.GetSpacing()[0])
                        j1 = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[1])
                        j2 = int(-1 * self.click2[1] // self.mov_img.GetSpacing()[1])
                        k1 = k1
                        k2 = k2

                    if self.click1[2] == self.click2[2] == "xz":
                        i1 = int(self.click1[0] // self.mov_img.GetSpacing()[0])
                        i2 = int(self.click2[0] // self.mov_img.GetSpacing()[0])
                        j1 = j1
                        j2 = j2
                        k1 = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[2])
                        k2 = int(-1 * self.click2[1] // self.mov_img.GetSpacing()[2])

                    if self.click1[2] == self.click2[2] == "yz":
                        i1 = i1
                        i2 = i2
                        j1 = int(self.click1[0] // self.mov_img.GetSpacing()[1])
                        j2 = int(self.click2[0] // self.mov_img.GetSpacing()[1])
                        k1 = int(-1 * self.click1[1] // self.mov_img.GetSpacing()[2])
                        k2 = int(-1 * self.click2[1] // self.mov_img.GetSpacing()[2])

                    # Get coordinates in physical space
                    xyz1 = np.array(self.mov_img.TransformIndexToPhysicalPoint([i1, j1, k1]))
                    xyz2 = np.array(self.mov_img.TransformIndexToPhysicalPoint([i2, j2, k2]))

                    # calculate transformation and update attributes and plots
                    self.delta = xyz1 - xyz2
                    print(f"delta: {self.delta}")
                    transform = sitk.TranslationTransform(3)
                    transform.SetOffset(xyz1 - xyz2)
                    self.mov_img = sitk.Resample(self.mov_img,
                                                 transform=transform,
                                                 interpolator=sitk.sitkLinear,
                                                 defaultPixelValue=0,
                                                 outputPixelType=self.mov_img.GetPixelIDValue())
                    self.update_plot()
                    if self.transform2 is None:
                        self.transform2 = transform
                    else:
                        self.transform2 = sitk.CompositeTransform([self.transform2, transform])

                else:
                    print("To align the brains, two clicks must occur on the same figure and axis. Resetting Clicks.")

                # reset the clicks
                self.click1 = None
                self.click2 = None

    def on_key(self, event):
        if event.key == "enter":
            if self.transform1 is None:
                self.transform1 = sitk.AffineTransform(3)
            if self.transform2 is None:
                self.transform2 = sitk.AffineTransform(3)
            self.transform = sitk.CompositeTransform([self.transform1, self.transform2])
            print("\nCentering completed.")
            print(f"Fix image center: {self.i, self.j, self.k}")
            print(f"Mov image center: {self.l, self.m, self.n}")
            print(f"Transform: {self.transform.GetParameters()}\n")
            plt.close(self.fig1)
            plt.close(self.fig2)

        if event.key == "r":
            """Reset the image by first inverting the transformation, and then setting the central slice."""
            print("Resetting image.")
            inverse_transform = self.transform.GetInverse()
            self.transform = None  # the transform must be reset
            self.mov_img = sitk.Resample(self.mov_img,
                                         transform=inverse_transform,
                                         interpolator=sitk.sitkLinear,
                                         defaultPixelValue=0,
                                         outputPixelType=self.mov_img.GetPixelIDValue())
            self.i = self.i0
            self.j = self.j0
            self.k = self.k0
            self.l = self.l0
            self.m = self.m0
            self.n = self.n0
            self.update_plot()

        if event.key == "up":
            print("Upscale atlas by a factor 1.1x")
            rescale = sitk.ScaleTransform(3)
            scale_factors = (0.9, 0.9, 0.9)
            rescale.SetScale(scale_factors)
            # Apply the 3D scale transform to the image
            self.mov_img = sitk.Resample(self.mov_img,
                                         transform=rescale,
                                         interpolator=sitk.sitkLinear,
                                         defaultPixelValue=0,
                                         outputPixelType=self.mov_img.GetPixelIDValue())
            self.update_transform(rescale)
            self.update_plot()

        if event.key == "down":
            print("Downscale atlas by a factor 0.9x")
            rescale = sitk.ScaleTransform(3)
            scale_factors = (1.1, 1.1, 1.1)
            rescale.SetScale(scale_factors)
            # Apply the 3D scale transform to the image
            self.mov_img = sitk.Resample(self.mov_img,
                                         transform=rescale,
                                         interpolator=sitk.sitkLinear,
                                         defaultPixelValue=0,
                                         outputPixelType=self.mov_img.GetPixelIDValue())
            self.update_transform(rescale)
            self.update_plot()

    def update_transform(self, transform):
        """Update the transform attribute"""
        if self.transform is None:
            self.transform = transform
        else:
            self.transform = sitk.CompositeTransform([self.transform, transform])




