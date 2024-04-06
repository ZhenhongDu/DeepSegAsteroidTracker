import os
import numpy as np
import image_sim as isim
from tifffile import imwrite


# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)


# simulation parameters
width_y = 1024
width_x = 1024

frame_num = 6

# camaera parameter
gain = 1.0
exposure = 15.0
dark = 0.1
sky_counts = 20
bias_level = 200
read_noise_electrons = 10

# star parameter
large_star_num = 5
middle_star_num = 120
small_star_num = 40

min_star_intensity = 10
max_star_intensity = 400

# asteroid parameter
asteroid_pos_x1 = 330
asteroid_pos_y1 = 480
asteroid_direction1 = 60 #angle,0-360
asteroid_velocity1 = 15

asteroid_pos_x2 = 800
asteroid_pos_y2 = 500
asteroid_direction2 = 10 #angle,0-360
asteroid_velocity2 = 25

asteroid_pos_x3 = 630
asteroid_pos_y3 = 150
asteroid_direction3 = 130 #angle,0-360
asteroid_velocity3 = 18

asteroid_intensity = 40 # intensity, related to SNR
asteroid_fwhm = 2

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
    asteroid1 = isim.asteroid(image, x_0=asteroid_pos_x1, y_0=asteroid_pos_y1, theta_0=asteroid_direction1,
                             v_0=asteroid_velocity1, t=time_point,
                             fluxes=asteroid_intensity, fwhm=asteroid_fwhm,
                             random_change=False)

    asteroid2 = isim.asteroid(image, x_0=asteroid_pos_x2, y_0=asteroid_pos_y2, theta_0=asteroid_direction2,
                             v_0=asteroid_velocity2, t=time_point,
                             fluxes=asteroid_intensity, fwhm=asteroid_fwhm,
                             random_change=False)

    asteroid3 = isim.asteroid(image, x_0=asteroid_pos_x3, y_0=asteroid_pos_y3, theta_0=asteroid_direction3,
                             v_0=asteroid_velocity3, t=time_point,
                             fluxes=asteroid_intensity, fwhm=asteroid_fwhm,
                             random_change=False)
    # gt image
    gt_stack[time_point, :, :] = stars_1 + stars_2 + stars_3 + asteroid1 + asteroid2 + asteroid3
    # noisy image
    noisy_stack[time_point, :, :] = (bias_only + noise_only + dark_only +
                                    flat * (sky_only + stars_1 + stars_2 + stars_3 + asteroid1 + asteroid2 + asteroid3)
                                    )

imwrite('/mnt/data/asteroid/simulation_results/simulate_multi_asteroid_data1.tif', noisy_stack)
imwrite('/mnt/data/asteroid/simulation_results/gt_multi_asteroid_data1.tif', gt_stack)