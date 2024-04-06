import os
import numpy as np
import image_sim as isim
from tifffile import imwrite
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch

# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)


# simulation parameters
width_y = 1024
width_x = 1024

frame_num = 1

# camaera parameter
gain = 1.0
exposure = 15.0
dark = 0.1
sky_counts = 20
bias_level = 200
read_noise_electrons = 10

# star parameter
large_star_num = 5
middle_star_num = 400
small_star_num = 40

min_star_intensity = 10
max_star_intensity = 400

# asteroid parameter
asteroid_pos_x = 330
asteroid_pos_y = 480
asteroid_direction = 60 #angle,0-360
asteroid_velocity = 15
asteroid_intensity = 45 # intensity, related to SNR
asteroid_fwhm = 3

# start simulation
image = np.zeros([width_y, width_x])

# sensitive map
flat = isim.sensitivity_variations(image)

# larger star
stars_1 = isim.stars(image, large_star_num, fwhm_min=4, fwhm_max=5, fluxes_min=min_star_intensity, fluxes_max=max_star_intensity)
# middle star
stars_2 = isim.stars(image, middle_star_num, fwhm_min=1.8, fwhm_max=2.5, fluxes_min=min_star_intensity, fluxes_max=max_star_intensity)
# small star
stars_3 = isim.stars(image, small_star_num, fwhm_min=1.5, fwhm_max=1.8,  fluxes_min=min_star_intensity, fluxes_max=max_star_intensity)

gt_stack = np.zeros((frame_num, width_y, width_x), dtype=np.float16)
noisy_stack = np.zeros((frame_num, width_y, width_x), dtype=np.float16)


for time_point in range(0, frame_num):
    print("Simulating star image at timepoint {}...".format(time_point))
    # calculate noise distribution
    bias_only = isim.bias(image, bias_level, realistic=False)
    dark_only = isim.dark_current(image, dark, exposure, gain=gain, hot_pixels=False)
    sky_only = isim.sky_background(image, sky_counts, gain=gain)
    noise_only = isim.read_noise(image, read_noise_electrons, gain=gain)
    # simulate asteroid
    asteroid = isim.asteroid(image, x_0=asteroid_pos_x, y_0=asteroid_pos_y, theta_0=asteroid_direction,
                             v_0=asteroid_velocity, t=time_point,
                             fluxes=asteroid_intensity, fwhm=asteroid_fwhm,
                             random_change=False)
    # gt image
    gt_image = stars_1 + stars_2 + stars_3 + asteroid
    # noisy image
    noisy_image = bias_only + noise_only + dark_only + flat * (sky_only + stars_1 + stars_2 + stars_3 + asteroid)

    # bkg_estimator = MedianBackground()
    # bkg = Background2D(noisy_image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)


    seg_threshold = 5
    seg_res = detect_sources(gt_image, seg_threshold, npixels=8)
    binary_seg = seg_res.data.astype('bool')
    star_area = np.sum(binary_seg)
    star_perc = star_area / (width_y*width_x)
    print(star_perc)



    plt.figure()
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(gt_image, cmap='gray')
    plt.show()


    plt.figure()
    plt.imshow(seg_res.data.astype('bool'), cmap='gray')
    plt.show()
# imwrite('/mnt/data/asteroid/simulation_results/diff_SNRs/simulate_data_SNR0s.tif', noisy_stack)
# imwrite('/mnt/data/asteroid/simulation_results/diff_SNRs/gt_data_SNR0s.tif', gt_stack)
