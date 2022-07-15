import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import cm
from matplotlib.colors import ListedColormap
from photutils.aperture import SkyCircularAperture, aperture_photometry

from astropy import stats

warnings.filterwarnings("ignore")


def zscale(img):
    '''
    zscale转换器
    '''
    norm = ImageNormalize(img, interval=ZScaleInterval(),
                          stretch=LinearStretch())
    return norm

def gamma_correction(colorbar, gamma=1.0):
    '''
    对colorbar做gamma校正
    '''
    default = cm.get_cmap(colorbar, 256)
    de_color = default(np.linspace(0, 1, 256))
    newcolors = de_color.copy()
    for i in range(256):
        newcolors[i, :] = de_color[int(255 * np.power(i/255, gamma)), :]
    return ListedColormap(newcolors)

def plot_beam(ax, header, xy=(2,2)):
    import matplotlib.patches as mpatches
    BMAJ = 3600. * header["BMAJ"] # [arcsec]
    BMIN = 3600. * header["BMIN"] # [arcsec]
    BPA =  header["BPA"] # degrees East of North
    print('BMAJ: {:.3f}", BMIN: {:.3f}", BPA: {:.2f} deg'.format(BMAJ, BMIN, BPA))
    # However, to plot it we need to negate the BPA since the rotation is the opposite direction
    # due to flipping RA.
    ax.add_artist(mpatches.Ellipse(xy=xy, width=BMIN, height=BMAJ, angle=-BPA, facecolor="none", color='white',linewidth=3))

class AstroImage:
    def __init__(self, url, ext=0, mode='file', verbose=False, **kwargs):
        '''
        初始化AstroImage
        '''
        self.url = url
        self.ext = ext
        if(mode == 'file'):
            self.open_fits_image()
            # 判断是否有多余的维度
            if((self.wcs.wcs.naxis != 2) & verbose):
                print(
                    f"This is not a 2D image.\nThis image has {self.wcs.wcs.naxis} axises: {self.wcs.wcs.ctype}.\nTry to auto drop axises.")
            while((self.wcs.wcs.naxis != 2) and (self.wcs.array_shape[0] == 1)):
                self.hdu = self.hdu[0]
                self.wcs = self.wcs.dropaxis(dropax=self.wcs.wcs.naxis-1)
                self.hdr.update(self.wcs.to_header())
            if((self.wcs.wcs.naxis == 2) & verbose):
                print(
                    f"Drop axises success! Now image has {self.wcs.wcs.naxis} axises: {self.wcs.wcs.ctype}.")
            else:
                Exception(
                    f"Drop axixes ERROR!\nThis image still has {self.wcs.wcs.naxis} axises: {self.wcs.wcs.ctype}.")

            # Pixel scale
            self.pixel_scale = proj_plane_pixel_scales(self.wcs) * 3600
        elif(mode == 'data'):
            self.hdu = kwargs['hdu']
            self.hdr = kwargs['hdr']
            self.wcs = kwargs['wcs']
            self.pixel_scale = kwargs['pixel_scale']

    def open_fits_image(self):
        '''
        初始化时候用到的读取文件的函数
        '''
        with fits.open(self.url) as hdulist:
            self.hdu = hdulist[self.ext].data
            self.hdr = hdulist[self.ext].header
            self.wcs = WCS(self.hdr)

    def cutout(self, coord, box):
        '''
        截图的函数，coord输入的ra，dec的数组；box输入的矩形长和宽，单位是arcsec
        '''
        coord = np.array([coord], dtype=np.float64)
        x, y = self.wcs.wcs_world2pix(coord, 0)[0]
        box = np.array(box) / np.array(self.pixel_scale)
        hdu_crop = Cutout2D(self.hdu, (x, y), box,
                            wcs=self.wcs, mode='partial')
        wcs_crop = hdu_crop.wcs
        self.hdr.update(wcs_crop.to_header())
        return AstroImage(url="", mode="data", hdu=hdu_crop.data, hdr=self.hdr, wcs=wcs_crop, pixel_scale=self.pixel_scale)

    def cutout_pixel(self, coord, box):
        '''
        截图的函数，coord输入的ra，dec的数组；box输入的矩形长和宽，单位是pixel
        '''
        coord = np.array([coord], dtype=np.float64)
        x, y = self.wcs.wcs_world2pix(coord, 0)[0]
        box = np.array(box)
        hdu_crop = Cutout2D(self.hdu, (x, y), box,
                            wcs=self.wcs, mode='partial')
        wcs_crop = hdu_crop.wcs
        self.hdr.update(wcs_crop.to_header())
        return AstroImage(url="", mode="data", hdu=hdu_crop.data, hdr=self.hdr, wcs=wcs_crop, pixel_scale=self.pixel_scale)

    def preview(self, color_map='gray_r', gamma=1.0, **kwargs):
        '''
        预览图像，只做预览用，科学图像应手动画。
        '''
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100,
                               subplot_kw={'projection': self.wcs})
        norm = zscale(self.hdu)
        if(gamma == 1):
            img = ax.imshow(self.hdu, cmap=color_map,
                            norm=norm, origin='lower')
        else:
            img = ax.imshow(self.hdu, cmap=gamma_correction(
                color_map, gamma=gamma), norm=norm, origin='lower')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        fig.colorbar(img, ax=ax, shrink=0.8)
        return fig, ax

    def save(self, url):
        '''
        保存为fits文件
        '''
        hdu = fits.PrimaryHDU(self.hdu, header=self.hdr)
        hdu.writeto(url, overwrite=True)

    def __repr__(self):
        '''
        在输出的时候显示一些信息
        '''
        fig, ax = self.preview()
        plt.show()
        return f"2D {self.wcs.array_shape} AstroImage"

    def mask_blank(self, threshold=10000):
        '''
        Masking the probably border or blank region
        '''
        values, counts = np.unique(self.hdu, return_counts=True)
        critical_counts = np.mean(counts) + 5*np.std(counts) + threshold
        print("critical counts = ", critical_counts)
        mode_values = values[counts > critical_counts]
        mode_counts = counts[counts > critical_counts]
        for i, m in enumerate(mode_values):
            print(mode_counts[i], " pixels' value is ", m)
            outregion = (self.hdu == m)
            self.hdu[outregion] = np.nan
        return self

    def photometry(self, coord, r=1, mode='pixel', zeropoint=0, plot=False):
        '''
        圆形口径测光
        '''
        if(mode == 'pixel'):
            positions = self.wcs.pixel_to_world(*coord)
        elif(mode == 'sky'):
            positions = SkyCoord(
                ra=coord[0]*u.degree, dec=coord[1]*u.degree, frame='fk5')
        aperture = SkyCircularAperture(positions, r=r*u.arcsec)
        counts = aperture_photometry(self.hdu, aperture, wcs=self.wcs)[
            'aperture_sum'].value[0]
        mag = -2.5 * np.log10(counts) + zeropoint

        if(not plot):
            return mag, aperture
        else:
            fig, ax = self.preview()
            ap_patches = aperture.to_pixel(self.wcs).plot(color='red', lw=2)
            ax.text(0.05, 0.9, f"{mag=:.2f}\n{r=:.2f}''",
                    transform=plt.gca().transAxes, color='red')
            return mag, aperture, fig, ax

    def sigma_clipped_stats(self, sigma=2, maxiters=5, **kwargs):
        '''
        圆形口径测光
        '''
        mean, median, stddev = stats.sigma_clipped_stats(self.hdu, sigma=sigma, maxiters=maxiters, **kwargs)
        return mean, median, stddev

    def mag_err(self, mag, zeropoint, area):
        '''
        计算magnitude的误差
        '''
        mean, median, stddev = self.sigma_clipped_stats()
        flux_err = stddev * np.sqrt(area)
        mag_err = 2.5/np.log(10) * flux_err / 10**((mag-zeropoint)/(-2.5))
        return mag_err