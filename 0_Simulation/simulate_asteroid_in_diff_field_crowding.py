import os
import numpy as np
import image_sim as isim
from tifffile import imwrite


# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)

# simulation parameters
width_y = 500
width_x = 500

frame_num = 10

# camaera parameter
gain = 1.0
exposure = 15.0
dark = 0.1
sky_counts = 20
bias_level = 200
read_noise_electrons = 10

# star parameter
large_star_num = 3
middle_star_num = 170
small_star_num = 7

min_star_intensity = 10
max_star_intensity = 400

# asteroid parameter
asteroid_pos_x = 200
asteroid_pos_y = 200


asteroid_direction = 60 #angle,0-360
asteroid_velocity = 15
asteroid_intensity = 40 # intensity, related to SNR
asteroid_fwhm = 3

save_path = '/data/Data_zhenhong/asteroid/simulation_results/diff_field_crowding/'
save_name_noise = save_path + 'simulate_data_star_num_180.tif'
save_name_gt = save_path + 'gt_data_star_num_180.tif'

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
    gt_stack[time_point, :, :] = stars_1 + stars_2 + stars_3 + asteroid
    # noisy image
    noisy_stack[time_point, :, :] = bias_only + noise_only + dark_only + flat * (sky_only + stars_1 + stars_2 + stars_3 + asteroid)

imwrite(save_name_noise, noisy_stack)
imwrite(save_name_gt, gt_stack)
