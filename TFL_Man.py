import DetectTFL
import Calculations
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('model.h5')

def work_on_frame(prev_frame, curr_frame,pkl_path):
    if prev_frame != 0:
        prev_frame.traffic_light = run_part1_and_part2_wo_plot(prev_frame)
        curr_frame.traffic_light, axs = run_part1_and_part2_and_plot(curr_frame)
    else:
        curr_frame.traffic_light, axs = run_part1_and_part2_and_plot(curr_frame)
    # PART3
    if prev_frame != 0:
       Calculations.read_data_and_run(prev_frame, curr_frame, axs, pkl_path)
    plt.show()

def run_part1_and_part2_and_plot(frame):
    # PART 1 AND PART 2
    predicted_x, predicted_y = [], []
    #Plot all candidates and returns them
    list_of_candidates, new_x, new_y, axs = DetectTFL.all_tfl_candidates(frame.path, "image_name")
    num_of_correct_predict = 0
    for i in range(len(list_of_candidates)):
        list_of_candidates[i] = list_of_candidates[i].reshape(1, 81, 81, 3)
        Part2_results = model.predict(list_of_candidates[i])[0]
        if Part2_results[0] < 0.99:
            num_of_correct_predict += 1
            predicted_x.append(new_x[i])
            predicted_y.append(new_y[i])
    #Prints all predicted points
    DetectTFL.printPrediction(frame.path, predicted_x, predicted_y, axs)
    return np.column_stack((predicted_x, predicted_y)),axs

def run_part1_and_part2_wo_plot(frame):
    # PART 1 AND PART 2
    predicted_x, predicted_y = [], []
    # Plot all candidates and returns them
    list_of_candidates, new_x, new_y = DetectTFL.all_tfl_candidates_wo_plotting(frame.path, "image_name")
    num_of_correct_predict = 0
    for i in range(len(list_of_candidates)):
        list_of_candidates[i] = list_of_candidates[i].reshape(1, 81, 81, 3)
        Part2_results = model.predict(list_of_candidates[i])[0]
        if Part2_results[0] < 0.99:
            num_of_correct_predict += 1
            predicted_x.append(new_x[i])
            predicted_y.append(new_y[i])
    # Prints all predicted points
    return np.column_stack((predicted_x, predicted_y))
