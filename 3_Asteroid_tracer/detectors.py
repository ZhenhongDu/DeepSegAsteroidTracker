# import the necessary packages
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import math
from utils import calculate_single_slope

class AsteroidTracker():
    # The initialization parameters
    def __init__(self):
        """
        star_id: id number of stars (containing both star and asteroid)
        asteroid_id: id number of asteroid
        current_frame: frame number of current object

        object: object is an OrderedDict to store the star_id
        and location information of moving stars.
        ------
        asteroid: If an object is living in three consecutive frame,
        the object is termed as asteroid when it disappeared.

        """
        self.star_id = 0
        self.asteroid_id = 0
        self.current_frame = 0

        self.objects = OrderedDict()
        self.asteroid = OrderedDict()
        self.disappeared_times = OrderedDict()

        self.wrong_flag = False
        self.MAX_disappeared_time = 2
        self.asteroid_criterion_frame = 3
        self.MAX_Predict_dist_error = 6

        self.asteroid_slope = {}

    def register(self, df_input):
        """
        This function is used to register new moving stars that
        were appeared in the first time.
        We store two kind of information:
        * star_id: indicating which one we want to track
        * location information: a DataFrame that stores the location information of
        the corresponding moving object

        :param df_input: a dataframe containing location information
        :return:
        """
        df_init = pd.DataFrame(columns=['centroid-0', 'centroid-1', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3'])

        self.objects[self.star_id] = df_init._append(df_input, ignore_index=True)
        self.objects[self.star_id]['prev_dist'] = 0
        self.objects[self.star_id]['current_frame'] = self.current_frame

        self.disappeared_times[self.star_id] = 0

        self.star_id += 1

    def deregister(self, id_num):
        """
        Deregister the disappeared stars
        if an object exists more than n frame --> asteroid result
        (n = self.asteroid_criterion_frame)
        :param id_num: id number of the star that you want to deregister
        :return:
        """
        # delete the object with specific id num >=self.asteroid_criterion_frame
        if id_num not in list(self.objects.keys()):
            return

        if self.objects[id_num].shape[0] >= self.asteroid_criterion_frame:
            self.asteroid[self.asteroid_id] = self.objects[id_num]
            self.asteroid_id += 1

        del self.objects[id_num]

    # Tracking the asteroid
    def track_obj(self, id_num, df_obj, prev_dist, time_interval):
        # add two columns
        df_obj['prev_dist'] = prev_dist

        # predict distance
        if self.objects[id_num].shape[0] >= 2:
            cord1 = (self.objects[id_num].iloc[-1]['centroid-0'], self.objects[id_num].iloc[-1]['centroid-1'])
            cord2 = (self.objects[id_num].iloc[-2]['centroid-0'], self.objects[id_num].iloc[-2]['centroid-1'])

            prev_time_interval = self.objects[id_num].iloc[-1]['current_frame'] - self.objects[id_num].iloc[-2]['current_frame']

            v_x = (cord1[0] - cord2[0]) / prev_time_interval
            v_y = (cord1[1] - cord2[1]) / prev_time_interval

            pred_x = cord1[0] + v_x * time_interval
            pred_y = cord1[1] + v_y * time_interval

            pred_cord = (pred_x, pred_y)
            input_cord = (df_obj['centroid-0'], df_obj['centroid-1'])

            err_dist = math.sqrt((input_cord[0] - pred_cord[0]) ** 2 + (input_cord[1] - pred_cord[1]) ** 2)

            if err_dist > self.MAX_Predict_dist_error:
                self.wrong_flag = True

        self.objects[id_num] = self.objects[id_num]._append(df_obj, ignore_index=True)


    def finish_mission(self):
        obj_ids = list(self.objects.keys())
        for obj_id in obj_ids:
            if self.objects[obj_id].shape[0] > 2:
                self.asteroid[self.asteroid_id] = self.objects[obj_id]
                self.asteroid_id += 1

    def check_slope(self):
        for key, df in self.asteroid.items():
            slope = calculate_single_slope(df, 'centroid-0', 'centroid-1')
            self.asteroid_slope[key] = slope


    # calculate the distance between two coordinates
    def get_input_centroids(self, df):
        """
        Get the centroids information from input DataFrame
        :param df:
        :return:
        """
        # data should be a dataframe
        centroids = np.zeros((df.shape[0], 2), dtype='float32')
        for index, row in df.iterrows():
            centroids[index] = (row['centroid-0'], row['centroid-1'])
        return centroids

    def update(self, df):
        """
        Updating tracking information with input moving object
        :param df: a DataFrame contains input moving object
        :return:
        """
        ############
        # STEP1: Check is there a moving object in the input frame.
        ############
        # If there are no moving object in current frame
        # We should increase the disappeared time of all existing object
        # If one object has disappeared more than MAX_disapeared_time
        # We should deregister it.
        if df.shape[0] == 0:
            for obj_ID in list(self.disappeared_times.keys()):
                self.disappeared_times[obj_ID] += 1
                if self.disappeared_times[obj_ID] > self.MAX_disappeared_time:
                    self.deregister(obj_ID)
            return

        ############
        # STEP2: If there a tracking object?
        # NO --> we should register all moving object in the input frame
        # YES --> We should try to track them with input moving stars
        ############
        if len(self.objects) == 0:
            for i in range(df.shape[0]):
                self.register(df.iloc[i])
        else:
            # calculate the distance between existing objects
            # and current input centroids
            self.current_frame = df['current_frame'].iloc[0]

            obj_ids = list(self.objects.keys())
            asteroid_ids = []

            input_cor = self.get_input_centroids(df)

            ############
            # STEP3: Moving object tracing strategy
            # Fist, calculate the distance between moving obj
            # and existing obj --> get matched moving stars
            # ------
            # Case1:
            # if moving object <= existing object
            # disappeared_time +=1
            # if disappeared_time > MAx_disappeared_time
            # deregister the existing obj
            # register unmatched moving object
            # Case2:
            # if moving object > existing object
            # register unmatched moving object
            # -----
            ############

            matched_existing_objs_id = []
            matched_moving_objs_id = []

            for idx in obj_ids:
                # for a specific star
                # calculate the dist between this star and current input coordinate
                obj = self.objects[idx].iloc[-1]
                aim_cor = np.asarray([obj['centroid-0'], obj['centroid-1']])

                # calculate the distance between obj[idx] and current frame
                dist_matrix = cdist(aim_cor.reshape(1, 2), input_cor)
                # delete fixed points
                dist_matrix[dist_matrix < 1.8] = 100000
                # find the min value of each row
                min_dist = dist_matrix.min(axis=1)
                min_index = dist_matrix.argmin(axis=1)

                time_interval = self.current_frame - obj['current_frame']

                # track the object using min_dist
                if obj['prev_dist'] == 0:
                    if time_interval == 1:
                        if 5 < min_dist < 70:
                            self.track_obj(idx, df.iloc[min_index], min_dist, time_interval)
                            if not self.wrong_flag:
                                asteroid_ids.append(min_index)
                                matched_existing_objs_id.append(idx)
                                matched_moving_objs_id.append(min_index)
                    if time_interval >= 2:
                        # second row
                        if 10 < min_dist < 150:
                            self.track_obj(idx, df.iloc[min_index], min_dist, time_interval)
                            if not self.wrong_flag:
                                asteroid_ids.append(min_index)
                                matched_existing_objs_id.append(idx)
                                matched_moving_objs_id.append(min_index)

                else:
                    if 0.75 * obj['prev_dist'] * time_interval < min_dist < 1.25 * obj['prev_dist'] * time_interval:
                        self.track_obj(idx, df.iloc[min_index], min_dist/time_interval, time_interval)
                        if not self.wrong_flag:
                            asteroid_ids.append(min_index)
                            matched_existing_objs_id.append(idx)
                            matched_moving_objs_id.append(min_index)

                if self.wrong_flag:
                    del self.objects[idx]
                    self.wrong_flag = False

            existing_objs_num = len(self.objects)
            moving_stars_num = len(df)
            # Case1:
            # if moving object <= existing object
            # disappeared_time +=1
            # if disappeared_time > MAX_disappeared_time
            # deregister the existing obj
            if moving_stars_num <= existing_objs_num:
                for obj_ID in list(self.disappeared_times.keys()):
                    if obj_ID not in matched_existing_objs_id:
                        self.disappeared_times[obj_ID] += 1
                        if self.disappeared_times[obj_ID] > self.MAX_disappeared_time:
                            self.deregister(obj_ID)

            # register unmatched moving object
            # Case2:
            # if moving object > existing object
            # register unmatched moving object
            for moving_star_ID in range(df.shape[0]):
                if moving_star_ID not in matched_moving_objs_id:
                    self.register(df.iloc[moving_star_ID])
            # else:
            #     for moving_star_ID in range(df.shape[0]):
            #         if moving_star_ID not in matched_moving_objs_id:
            #             self.register(df.iloc[moving_star_ID])


        return self.objects



if __name__ == '__main__':
    """
    obj data:
       centroid-0  centroid-1 bbox-0 bbox-1 bbox-2 bbox-3  current_frame  prev_dist
    0  299.809524  199.833333    297    196    304    204              0          0
    """