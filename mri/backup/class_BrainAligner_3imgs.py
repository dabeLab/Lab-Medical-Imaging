import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import SimpleITK as sitk
from utilities import custom_colormap


class BrainAligner:
    """
    This class contains a set of methods and attributes that assist the user in aligning two images.
    The class assumes that the two images are registered in the same physical space - In an updated version this step
    could be included in this class.
    """

    def __init__(self, mri1: sitk.Image, mri2: sitk.Image, mri3: sitk.Image, atlas: sitk.Image):
        """
        The physical space is described by the (x, y, z) coordinate system
        The fixed image space is described by the (i, j, k) coordinate system
        The moving image space is described by the (l, m, n) coordinate system
        """

        self.fig = None
        self.ax = None

        self.atlas = atlas
        self.mri1 = mri1
        self.mri2 = mri2
        self.mri3 = mri3

        self.click1 = None
        self.click2 = None

        # initialize slice
        self.i0 = fix_img.GetSize()[0] // 2
        self.j0 = fix_img.GetSize()[1] // 2
        self.k0 = fix_img.GetSize()[2] // 2
        self.l0 = mov_img.GetSize()[0] // 2
        self.m0 = mov_img.GetSize()[1] // 2
        self.n0 = mov_img.GetSize()[2] // 2

        # current and previous_ slice index
        self.i = self.i0
        self.j = self.j0
        self.k = self.k0
        self.l = self.l0
        self.m = self.m0
        self.n = self.n0

        # delta for alignment
        self.delta_ijk = 0
        self.delta_lmn = 0
        # Note for the calculation of the final transformation
        # lmn1 = lmn0 + delta_lmn
        # ijk1 = ijk0 + delta_ijk
        # lmn1 - ijk1 = (lmn0 + delta_lmn)- (ijk0 + delta_ijk) = delta_lmn - delta_ijk
        # since lmn0 = ijk0

        # Transformation
        self.transform = None

    def execute(self):
        """The function creates a figure with xy, xz and yz slices of the MR image (fixed image), the brain atlas (moving image) and their
        overlay. First, the user is asked to mark the center of the brain, in the MR image and the atlas, by left-click. Then, the user can translate
        the brain atlas by rigth-click, so that the marked centers coincide. By pressing the up and down keys, the user can rescale the size of the
        brain atlas. By pressing return, the user accepts the alignment and the final transformation, which takes into account for both the
        selection of the image centers, and the translation of the brain atlas, is stored as attribute and the image closed.
        Note: this is the only solution that keeps the fixed image in its starting position."""
        # plot slices passing through the image volume center
        self.create_figure(self.atlas)
        self.create_figure(self.mri1)
        self.create_figure(self.mri2)
        self.create_figure(self.mri3)
        self.update_figure(np.array(self.fix_img.GetSize(), dtype=int) // 2,
                           np.array(self.mov_img.GetSize(), dtype=int) // 2)
        print("Left-click to mark images center.\n"
              "Right-click to align.\n"
              "Press Up/Down keys to up/down-scale brian atlas.\n"
              "Press Return to accept and store transformation.\n")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show(block=False)
        mplcursors.cursor(hover=True)
        plt.show()

    def create_figure(self, img: sitk.Image):
        """Create figure and axes for brain centering"""
        fig, ax = plt.subplots(2, 3)
        plt.show(block=False)

        for idx, ax in enumerate(ax.flatten()):
            ax.set_axis_off()

        ax[0, 0].set_title("xy - axial")
        ax[0, 1].set_title("xz - coronal")
        ax[0, 2].set_title("yz - sagittal")

        # 1st row - img -----------------------------------------------------------
        ax[0, 0].set_label("xy fix")
        ax[0, 0].imshow(X=np.zeros(shape=(img.GetSize()[0], img.GetSize()[1])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[0] * img.GetSpacing()[0],
                                - img.GetSize()[1] * img.GetSpacing()[1],
                                0],
                        vmin=0,
                        vmax=1)
        ax[0, 0].axhline(y=-img.GetSize()[1] // 2 * img.GetSpacing()[1], color='darkred', linestyle='--', linewidth=1)
        ax[0, 0].axvline(x=img.GetSize()[0] // 2 * img.GetSpacing()[0], color='darkred', linestyle='--', linewidth=1)

        ax[0, 1].set_label("xz fix")
        ax[0, 1].imshow(X=np.zeros(shape=(img.GetSize()[0], img.GetSize()[2])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[0] * img.GetSpacing()[0],
                                - img.GetSize()[2] * img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[0, 1].axhline(y=-img.GetSize()[2] // 2 * img.GetSpacing()[2], color='darkred', linestyle='--', linewidth=1)
        ax[0, 1].axvline(x=img.GetSize()[0] // 2 * img.GetSpacing()[0], color='darkred', linestyle='--', linewidth=1)

        ax[0, 2].set_label("yz fix")
        ax[0, 2].imshow(X=np.zeros(shape=(img.GetSize()[1], img.GetSize()[2])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[1] * img.GetSpacing()[1],
                                - img.GetSize()[2] * img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[0, 2].axhline(y=-img.GetSize()[2] // 2 * img.GetSpacing()[2], color='darkred', linestyle='--', linewidth=1)
        ax[0, 2].axvline(x=img.GetSize()[1] // 2 * img.GetSpacing()[1], color='darkred', linestyle='--', linewidth=1)

        # 2rd row - overlay ------------------------------------------------------------------
        ax[1, 0].set_label("xy ovl")
        ax[1, 0].imshow(X=np.zeros(shape=(img.GetSize()[0], img.GetSize()[1])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[0] * img.GetSpacing()[0],
                                - img.GetSize()[1] * img.GetSpacing()[1],
                                0],
                        vmin=0,
                        vmax=1)
        ax[1, 0].imshow(X=np.zeros(shape=(img.GetSize()[0], img.GetSize()[1])),
                        cmap=custom_colormap(),
                        extent=[0,
                                img.GetSize()[0] * img.GetSpacing()[0],
                                - img.GetSize()[1] * img.GetSpacing()[1],
                                0],
                        vmin=0,
                        vmax=1)
        ax[1, 0].axhline(y=-img.GetSize()[1] // 2 * img.GetSpacing()[1], color='darkred', linestyle='--', linewidth=1)
        ax[1, 0].axvline(x=img.GetSize()[0] // 2 * img.GetSpacing()[0], color='darkred', linestyle='--', linewidth=1)
        ax[1, 0].axhline(y=-img.GetSize()[1] // 2 * img.GetSpacing()[1], color='olive', linestyle='--', linewidth=1)
        ax[1, 0].axvline(x=img.GetSize()[0] // 2 * img.GetSpacing()[0], color='olive', linestyle='--', linewidth=1)

        ax[2, 1].set_label("xz ovl")
        ax[2, 1].imshow(X=np.zeros(shape=(img.GetSize()[0], img.GetSize()[2])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[0] * img.GetSpacing()[0],
                                - img.GetSize()[2] * img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[2, 1].imshow(X=np.zeros(shape=(self.mov_img.GetSize()[0], self.mov_img.GetSize()[2])),
                        cmap=custom_colormap(),
                        extent=[0,
                                self.mov_img.GetSize()[0] * self.mov_img.GetSpacing()[0],
                                - self.mov_img.GetSize()[2] * self.mov_img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[2, 1].axhline(y=-img.GetSize()[2] // 2 * img.GetSpacing()[2], color='darkred', linestyle='--', linewidth=1)
        ax[2, 1].axvline(x=img.GetSize()[0] // 2 * img.GetSpacing()[0], color='darkred', linestyle='--', linewidth=1)
        ax[2, 1].axhline(y=-self.mov_img.GetSize()[2] // 2 * self.mov_img.GetSpacing()[2], color='olive', linestyle='--', linewidth=1)
        ax[2, 1].axvline(x=self.mov_img.GetSize()[0] // 2 * self.mov_img.GetSpacing()[0], color='olive', linestyle='--', linewidth=1)

        ax[2, 2].set_label("yz ovl")
        ax[2, 2].imshow(X=np.zeros(shape=(img.GetSize()[1], img.GetSize()[2])),
                        cmap='gray',
                        extent=[0,
                                img.GetSize()[1] * img.GetSpacing()[1],
                                - img.GetSize()[2] * img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[2, 2].imshow(X=np.zeros(shape=(self.mov_img.GetSize()[1], self.mov_img.GetSize()[2])),
                        cmap=custom_colormap(),
                        extent=[0,
                                self.mov_img.GetSize()[1] * self.mov_img.GetSpacing()[1],
                                - self.mov_img.GetSize()[2] * self.mov_img.GetSpacing()[2],
                                0],
                        vmin=0,
                        vmax=1)
        ax[2, 2].axhline(y=-img.GetSize()[2] // 2 * img.GetSpacing()[2], color='darkred', linestyle='--', linewidth=1)
        ax[2, 2].axvline(x=img.GetSize()[1] // 2 * img.GetSpacing()[1], color='darkred', linestyle='--', linewidth=1)
        ax[2, 2].axhline(y=-self.mov_img.GetSize()[2] // 2 * self.mov_img.GetSpacing()[2], color='olive', linestyle='--', linewidth=1)
        ax[2, 2].axvline(x=self.mov_img.GetSize()[0] // 2 * self.mov_img.GetSpacing()[0], color='olive', linestyle='--', linewidth=1)

        plt.tight_layout()
        self.fig = fig
        self.ax = ax

    def update_figure(self, ijk: [int], lmn: [int]):
        """Update the displayed slices."""

        i = int(ijk[0])
        j = int(ijk[1])
        k = int(ijk[2])

        l = int(lmn[0])
        m = int(lmn[1])
        n = int(lmn[2])

        # -----------------------------------------------------------------------
        # Update fixed image - xy plane
        fix_img_xy_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], self.fix_img.GetSize()[1], 0], [0, 0, k])
        self.ax[0, 0].images[0].set_array(sitk.GetArrayFromImage(fix_img_xy_slice))
        self.ax[0, 0].lines[0].set_ydata(-j * self.fix_img.GetSpacing()[1])
        self.ax[0, 0].lines[1].set_xdata(+i * self.fix_img.GetSpacing()[0])

        # Update fixed image - xz plane
        fix_img_xz_slice = sitk.Extract(self.fix_img, [self.fix_img.GetSize()[0], 0, self.fix_img.GetSize()[2]], [0, j, 0])
        self.ax[0, 1].images[0].set_array(sitk.GetArrayFromImage(fix_img_xz_slice))
        self.ax[0, 1].lines[0].set_ydata(-k * self.fix_img.GetSpacing()[2])
        self.ax[0, 1].lines[1].set_xdata(+i * self.fix_img.GetSpacing()[0])

        # # Update fixed image - yz plane
        fix_img_yz_slice = sitk.Extract(self.fix_img, [0, self.fix_img.GetSize()[1], self.fix_img.GetSize()[2]], [i, 0, 0])
        self.ax[0, 2].images[0].set_array(sitk.GetArrayFromImage(fix_img_yz_slice))
        self.ax[0, 2].lines[0].set_ydata(-k * self.fix_img.GetSpacing()[2])
        self.ax[0, 2].lines[1].set_xdata(+j * self.fix_img.GetSpacing()[1])

        # -----------------------------------------------------------------------
        # Update moving image - xy plane
        mov_img_xy_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], self.mov_img.GetSize()[1], 0], [0, 0, n])
        self.ax[1, 0].images[0].set_array(sitk.GetArrayFromImage(mov_img_xy_slice))
        self.ax[1, 0].lines[0].set_ydata(-m * self.mov_img.GetSpacing()[1])
        self.ax[1, 0].lines[1].set_xdata(+l * self.mov_img.GetSpacing()[0])

        # Update moving image - xz plane
        mov_img_xz_slice = sitk.Extract(self.mov_img, [self.mov_img.GetSize()[0], 0, self.mov_img.GetSize()[2]], [0, m, 0])
        self.ax[1, 1].images[0].set_array(sitk.GetArrayFromImage(mov_img_xz_slice))
        self.ax[1, 1].lines[0].set_ydata(-n * self.mov_img.GetSpacing()[2])
        self.ax[1, 1].lines[1].set_xdata(+l * self.mov_img.GetSpacing()[0])

        # Update moving image - yz plane
        mov_img_yz_slice = sitk.Extract(self.mov_img, [0, self.fix_img.GetSize()[1], self.mov_img.GetSize()[2]], [l, 0, 0])
        self.ax[1, 2].images[0].set_array(sitk.GetArrayFromImage(mov_img_yz_slice))
        self.ax[1, 2].lines[0].set_ydata(-n * self.mov_img.GetSpacing()[2])
        self.ax[1, 2].lines[1].set_xdata(+m * self.mov_img.GetSpacing()[1])

        # -----------------------------------------------------------------------
        # Display fixed and moving image overlaid in Figure 2.
        self.ax[2, 0].images[0].set_array(sitk.GetArrayViewFromImage(fix_img_xy_slice))
        self.ax[2, 0].images[1].set_array(sitk.GetArrayViewFromImage(mov_img_xy_slice))
        self.ax[2, 0].lines[0].set_ydata(-j * self.fix_img.GetSpacing()[1])
        self.ax[2, 0].lines[1].set_xdata(+i * self.fix_img.GetSpacing()[0])
        self.ax[2, 0].lines[2].set_ydata(-m * self.mov_img.GetSpacing()[1])
        self.ax[2, 0].lines[3].set_xdata(+l * self.mov_img.GetSpacing()[0])
        self.ax[2, 1].images[0].set_array(sitk.GetArrayViewFromImage(fix_img_xz_slice))
        self.ax[2, 1].images[1].set_array(sitk.GetArrayViewFromImage(mov_img_xz_slice))
        self.ax[2, 1].lines[0].set_ydata(-k * self.fix_img.GetSpacing()[2])
        self.ax[2, 1].lines[1].set_xdata(+i * self.fix_img.GetSpacing()[0])
        self.ax[2, 1].lines[2].set_ydata(-n * self.mov_img.GetSpacing()[2])
        self.ax[2, 1].lines[3].set_xdata(+l * self.mov_img.GetSpacing()[0])
        self.ax[2, 2].images[0].set_array(sitk.GetArrayViewFromImage(fix_img_yz_slice))
        self.ax[2, 2].images[1].set_array(sitk.GetArrayViewFromImage(mov_img_yz_slice))
        self.ax[2, 2].lines[0].set_ydata(-k * self.fix_img.GetSpacing()[2])
        self.ax[2, 2].lines[1].set_xdata(+j * self.fix_img.GetSpacing()[1])
        self.ax[2, 2].lines[2].set_ydata(-n * self.mov_img.GetSpacing()[2])
        self.ax[2, 2].lines[3].set_xdata(+m * self.mov_img.GetSpacing()[1])
        self.fig.canvas.draw()

    def on_click(self, event):
        """
        Left click -> set center
        Right click -> align centers
        """""

        if event.inaxes:

            # Left click
            if event.button == 1:

                print(f'Click at ({event.xdata}, {event.ydata}) on {event.inaxes.get_label()}')

                if event.inaxes.get_label() == "xy fix":
                    self.i = int(event.xdata // self.fix_img.GetSpacing()[0])
                    self.j = int(-1 * event.ydata // self.fix_img.GetSpacing()[1])
                elif event.inaxes.get_label() == "xz fix":
                    self.i = int(event.xdata // self.fix_img.GetSpacing()[0])
                    self.k = int(-1 * event.ydata // self.fix_img.GetSpacing()[2])
                elif event.inaxes.get_label() == "yz fix":
                    self.j = int(event.xdata // self.fix_img.GetSpacing()[1])
                    self.k = int(-1 * event.ydata // self.fix_img.GetSpacing()[2])

                elif event.inaxes.get_label() == "xy mov":
                    self.l = int(event.xdata // self.mov_img.GetSpacing()[0])
                    self.m = int(-1 * event.ydata // self.mov_img.GetSpacing()[1])
                elif event.inaxes.get_label() == "xz mov":
                    self.l = int(event.xdata // self.mov_img.GetSpacing()[0])
                    self.n = int(-1 * event.ydata // self.mov_img.GetSpacing()[2])
                elif event.inaxes.get_label() == "yz mov":
                    self.m = int(event.xdata // self.mov_img.GetSpacing()[1])
                    self.n = int(-1 * event.ydata // self.mov_img.GetSpacing()[2])

                self.update_figure([self.i, self.j, self.k], [self.l, self.m, self.n])

            # Right click
            elif event.button == 3:

                print("Aligning images")

                # Get coordinates in physical space
                target = np.array(self.mov_img.TransformIndexToPhysicalPoint([self.i, self.j, self.k]))
                origin = np.array(self.mov_img.TransformIndexToPhysicalPoint([self.l, self.m, self.n]))

                # calculate transformation
                self.transform = sitk.TranslationTransform(3)
                self.transform.SetOffset(-(target - origin))
                self.mov_img = sitk.Resample(self.mov_img,
                                             transform=self.transform,
                                             interpolator=sitk.sitkLinear,
                                             defaultPixelValue=0,
                                             outputPixelType=self.mov_img.GetPixelIDValue())

                self.delta_lmn = self.mov_img.TransformIndexToPhysicalPoint([self.l - self.l0, self.m - self.m0, self.n - self.n0])
                self.delta_ijk = self.mov_img.TransformIndexToPhysicalPoint([self.i - self.i0, self.j - self.j0, self.k - self.k0])
                self.transform.SetOffset([y - x for (x, y) in zip(self.delta_ijk, self.delta_lmn)])

                # After translation, the centers of the moving images coincides with the center of the fixed image.
                # Therefore, the slices of the moving image to show are now passing through the new center (l,m,n) = (i,j,k).
                self.l = self.i
                self.m = self.j
                self.n = self.k
                self.update_figure([self.i, self.j, self.k], [self.l, self.m, self.n])
                self.click1 = None

    def on_key(self, event):
        if event.key == "enter":
            print("\nCentering completed.")
            print(f"Fix image center: {self.i, self.j, self.k}")
            print(f"Mov image center: {self.l, self.m, self.n}")
            print(f"Transform: {self.transform.GetParameters()}\n")
            plt.close(self.fig)

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
            self.update_figure([self.i, self.j, self.k], [self.l, self.m, self.n])

        if event.key == "up":
            print("Upscale atlas by a factor 1.1x")
            rescale = sitk.ScaleTransform(3)
            scale_factors = (0.9, 0.9, 0.9)
            rescale.SetScale(scale_factors)
            rescale.SetCenter(self.mov_img.TransformIndexToPhysicalPoint([self.l, self.m, self.n]))
            # Apply the 3D scale transform to the image
            self.mov_img = sitk.Resample(self.mov_img,
                                         transform=rescale,
                                         interpolator=sitk.sitkLinear,
                                         defaultPixelValue=0,
                                         outputPixelType=self.mov_img.GetPixelIDValue())
            self.update_transform(rescale)
            self.update_figure([self.i, self.j, self.k], [self.l, self.m, self.n])

        if event.key == "down":
            print("Downscale atlas by a factor 0.9x")
            rescale = sitk.ScaleTransform(3)
            scale_factors = (1.1, 1.1, 1.1)
            rescale.SetScale(scale_factors)
            rescale.SetCenter(self.mov_img.TransformIndexToPhysicalPoint([self.l, self.m, self.n]))
            # Apply the 3D scale transform to the image
            self.mov_img = sitk.Resample(self.mov_img,
                                         transform=rescale,
                                         interpolator=sitk.sitkLinear,
                                         defaultPixelValue=0,
                                         outputPixelType=self.mov_img.GetPixelIDValue())
            self.update_transform(rescale)
            self.update_figure([self.i, self.j, self.k], [self.l, self.m, self.n])

    def update_transform(self, transform):
        """Update the transform attribute"""
        if self.transform is None:
            self.transform = transform
        else:
            self.transform = sitk.CompositeTransform([self.transform, transform])

    # def update_slice_index(self):
    #     self.i = self.i_
    #     self.j = self.j_
    #     self.k = self.k_
    #     self.l = self.l_
    #     self.m = self.m_
    #     self.n = self.n_

