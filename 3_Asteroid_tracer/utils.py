import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from scipy.stats import linregress

def reverse_norm(im, low=0.2, high=99.8):
    max_ = np.percentile(im, high)
    min_ = np.percentile(im, low)
    im = np.clip(im, min_, max_)
    im = (im - min_) / (max_ - min_) * 255
    return im.astype(np.uint8)


def normalize_percentile(im, low=0.2, high=99.8):
    """Normalize the input 'im' by im = (im - p_low) / (p_high - p_low), whe
re p_low/p_high is the 'low'th/'high'th percentile of the im
    Params:
        -im  : numpy.ndarray
        -low : float, typically 0.2
        -high: float, typically 99.8
    return:
        normalized ndarray of which most of the pixel values are in [0, 1]
    """

    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    eps = 1e-10
    im = (im - min_v) / (max_v - min_v + eps)
    return im

def get_centroids(data):
    # label image regions
    # label_image = label(image)
    # regions = regionprops(label_image)
    # props = regionprops_table(label_image, properties=('centroid', 'bbox'))
    # data = pd.DataFrame(props)
    inputCentroids = np.zeros((data.shape[0], 2), dtype='float32')
    for index, row in data.iterrows():
        inputCentroids[index] = (row['centroid-0'], row['centroid-1'])

    return inputCentroids

def get_fixed_stars(prop1, prop2):
    data1 = pd.DataFrame(prop1)
    data2 = pd.DataFrame(prop2)
    coor1 = get_centroids(data1)
    coor2 = get_centroids(data2)
    # calculate the distance between two coordinates
    dist_matrix = cdist(coor1, coor2)
    # find the min value of each row
    min_dist = dist_matrix.min(axis=1)
    # rows = dist_matrix.argmin(axis=1) # index of min value
    potential_target = pd.DataFrame(columns=data1.columns)
    for i in range(len(min_dist)):
        # detect fixed stars
        if min_dist[i] < 3:
            insert_row = data1.iloc[[i]]
            potential_target = pd.concat([potential_target, insert_row], ignore_index=True)

    return potential_target


def get_moving_stars(prop, fixed_star, frame_num):
    data1 = pd.DataFrame(prop)
    data1['current_frame'] = frame_num
    coor1 = get_centroids(data1)
    coor2 = get_centroids(fixed_star)
    # calculate the distance between two coordinates
    dist_matrix = cdist(coor1, coor2)
    # find the min value of each row
    min_dist = dist_matrix.min(axis=1)
    # rows = dist_matrix.argmin(axis=1) # index of min value
    moving_stars_coor = pd.DataFrame(columns=data1.columns)

    for i in range(len(min_dist)):
        # detect moving stars
        if min_dist[i] >= 3:
            insert_row = data1.iloc[[i]]
            # .append was deprecated in Pandas2.0
            # moving_stars_coor = moving_stars_coor._append(insert_row, ignore_index=True)
            moving_stars_coor = pd.concat([moving_stars_coor, insert_row], ignore_index=True)

    return moving_stars_coor


def show_with_rect(image1, potential_target):
    # draw the box
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image1, cmap='gray')

    for index, row in potential_target.iterrows():
        minr, minc, maxr, maxc = potential_target.loc[index, ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']]
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def get_outer_points(df):
    cord1 = (df.iloc[-1]['centroid-0'], df.iloc[-1]['centroid-1'])
    cord2 = (df.iloc[-2]['centroid-0'], df.iloc[-2]['centroid-1'])
    post_cord = (cord1[0] * 2 - cord2[0], cord1[1] * 2 - cord2[1])

    cord3 = (df.iloc[1]['centroid-0'], df.iloc[1]['centroid-1'])
    cord4 = (df.iloc[0]['centroid-0'], df.iloc[0]['centroid-1'])
    pre_cord = (cord4[0] * 2 - cord3[0], cord4[1] * 2 - cord3[1])

    return pre_cord, post_cord


def fill_missing_frames(df):
    # Ensure the DataFrame is sorted by 'current_frame'
    df = df.sort_values(by='current_frame')

    # Identify all expected frames (from min to max)
    full_frames = range(df['current_frame'].min(), df['current_frame'].max() + 1)

    # Identify missing frames
    missing_frames = sorted(set(full_frames) - set(df['current_frame']))

    # Create a new DataFrame to hold the interpolated values
    new_rows = []

    for frame in missing_frames:
        # Get the rows before and after the missing frame
        before = df[df['current_frame'] < frame].iloc[-1]
        after = df[df['current_frame'] > frame].iloc[0]

        # Linear interpolation for each column
        interpolated_row = before + (after - before) * (frame - before['current_frame']) / (after['current_frame'] - before['current_frame'])
        interpolated_row['current_frame'] = frame
        new_rows.append(interpolated_row)

    # Combine the original DataFrame with the new rows and sort
    df_filled = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df_filled = df_filled.sort_values(by='current_frame').reset_index(drop=True)

    filled_df_index = find_interpolated_frame_indices(df_filled, df)
    return df_filled, filled_df_index

def find_interpolated_frame_indices(filled_df, original_df):
    # Find the 'current_frame' values that were not in the original DataFrame
    missing_frames = set(filled_df['current_frame']) - set(original_df['current_frame'])
    # Get the indices of these missing frames in the filled DataFrame
    interpolated_indices = filled_df[filled_df['current_frame'].isin(missing_frames)].index.tolist()

    return interpolated_indices

def calculate_single_slope(df, x_col, y_col):
    """
    Calculate slope of the
    :param df: DataFrame
    :param x_col: clown x
    :param y_col: clown y
    :return: slope of the coordinate
    """

    x = df[x_col].values
    y = df[y_col].values
    slope, _, _, _, _ = linregress(x, y)

    return slope


if __name__ == '__main__':
    # Test the function with multiple missing frames
    # Modify the example DataFrame to have multiple missing frames
    data_with_multiple_missing = {
        'centroid-0': [299.333333, 285.857143, 274.071429, 221.550000, 208.619048, 196.333333, 183.263158],
        'centroid-1': [200.333333, 207.071429, 214.857143, 244.850000, 252.047619, 260.125000, 266.315789],
        'bbox-0': [298, 284, 273, 220, 206, 194, 181],
        'bbox-1': [199, 205, 213, 243, 250, 258, 265],
        'bbox-2': [301, 289, 276, 225, 212, 200, 187],
        'bbox-3': [202, 210, 218, 248, 255, 263, 269],
        'current_frame': [0, 1, 2, 6, 7, 9, 10],
        'prev_dist': [0.000000, 15.066834, 14.125161, 15.568082, 14.799164, 14.703162, 14.462212]
    }

    df_with_multiple_missing = pd.DataFrame(data_with_multiple_missing)

    # Fill missing frames
    df_filled_multiple, index = fill_missing_frames(df_with_multiple_missing)

    print(df_with_multiple_missing)
    print(df_filled_multiple)
    print(index)