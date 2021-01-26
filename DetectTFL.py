try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")

def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    green_i_list, green_j_list = [], []
    red_i_list, red_j_list = [], []
    actual_image = c_image
    c_image = c_image.astype(float)
    c_image = c_image[:, :, [0]]
    kernel = np.array([[0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    kernel = np.resize(kernel, (5, 5, 1))
    kernel = kernel.astype(float)
    kernel = kernel - np.average(kernel)
    t = sg.convolve(c_image, kernel, mode='same')
    filterd_image = ndimage.maximum_filter(t, size=250)
    filterd_image = filterd_image == t
    green_j, green_i, z = np.where(filterd_image)
    for i in range(len(z)):
            green_pixel_color = actual_image[green_j[i], green_i[i], [1]]
            red_pixel_color = actual_image[green_j[i], green_i[i], [0]]
            if green_pixel_color > red_pixel_color:
                green_i_list.append(green_i[i])
                green_j_list.append(green_j[i])
            else:
                red_i_list.append(green_i[i])
                red_j_list.append(green_j[i])
    return red_i_list, red_j_list, green_i_list, green_j_list

def show_image_and_gt(image, objs, axs, fig_num=None):
    axs[0].imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()

def verify_tfl_images(red_x, red_y, green_x, green_y , image): # labeld_image
    tfl_positive = np.array([]).astype('uint8')
    tfl_nigative = np.array([]).astype('uint8')
    i = 0
    while i < len(red_x):
            temp_img = image[red_y[i] - 40:red_y[i] + 41, red_x[i] - 40 : red_x[i] + 41 ]
            if temp_img.shape == (81, 81, 3):
                tfl_nigative = np.append(tfl_nigative, temp_img).astype('uint8')
            i += 1
    i = 0
    while i < len(green_x):
            temp_img = image[green_y[i] - 40:green_y[i] + 41, green_x[i] - 40: green_x[i] + 41]
            if temp_img.shape == (81,81, 3):
                tfl_nigative = np.append(tfl_nigative, temp_img).astype('uint8')
            i += 1
    return tfl_positive, tfl_nigative

def show_croped_images(tfl_positive, tfl_nigative):
    w = 81
    h = 81
    fig = plt.figure()
    rows = 4
    cols = 4
    i = 0
    j = 1
    axes = []
    for tfl in tfl_positive:
        axes.append(fig.add_subplot(rows, cols, j))
        subplot_title = ("YES TFL")
        axes[-1].set_title(subplot_title)
        plt.imshow(tfl)
        j += 1
        i += 1
    i = 0
    for tfl in tfl_nigative:
        axes.append(fig.add_subplot(rows, cols, j))
        subplot_title = ("NO TFL")
        axes[-1].set_title(subplot_title)
        plt.imshow(tfl)
        i += 1
        j += 1
    fig.tight_layout()
    plt.show()

def save_tfl_as_binary_files(tfl_positive, tfl_nigative, image_name):
    tfl_positive.tofile(image_name + "positive.bin")
    tfl_nigative.tofile(image_name + "nigative.bin")
    pass

def test_find_tfl_lights(image_path, image_name, json_path=None, fig_num=None ) :
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    tfl_positive, tfl_nigative = verify_tfl_images(red_x, red_y, green_x, green_y, image)#labeld_image, image)
    show_croped_images(tfl_positive, tfl_nigative)
    return tfl_positive, tfl_nigative

def save_tfl_data_as_binary(data): #labeled):
    data.tofile("data.bin")

def main(argv=None):
    DATA_PATH = "/Users/nalba/PycharmProjects/MobilyeProject/DataSet/leftImg8bit_trainvaltest/leftImg8bit"
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    args_original_photos = parser.parse_args(argv)
    if args_original_photos.dir is None:
        args_original_photos.dir = DATA_PATH + "/test"
    orignal_photos_folders = glob.glob(os.path.join(args_original_photos.dir, '*'))
    positive = np.array([]).astype('uint8')
    negative = np.array([]).astype('uint8')
    for folder in orignal_photos_folders:
        folder_name = folder.split('\\')[-1]
        flist = glob.glob(os.path.join(folder, '*_leftImg8bit.png')) # images from this folder
        for image in flist:
            image_file_name = image.split('\\')[-1].split("leftImg8bit.png")[0] + "gtFine_labelIds.png"
            po, ne = test_find_tfl_lights(image,image_file_name)
            positive = np.append(positive, po)
            negative = np.append(negative, ne)
    data = np.append(positive,negative).astype('uint8')
    print(positive.size, negative.size/(81*81*3))
    save_tfl_data_as_binary(negative)

#FIRST PART
#receives image and shows all candidates
def all_tfl_candidates(image_path,image_name, json_path=None, fig_num=None):
    frame_path = image_path.split("\\")[-1]
    fig, axs = plt.subplots(3)
    fig.suptitle(frame_path)
    original=Image.open(image_path)
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, axs, fig_num)
    red_x, red_y, green_x, green_y = find_all_tfl_candidates(image, some_threshold=42)
    axs[0].plot(red_x, red_y, 'ro', color='r', markersize=4)
    axs[0].plot(green_x, green_y, 'ro', color='g', markersize=4)
    axs[0].set_ylabel('Candidates')
    x, y, z = verify_candidate_images(red_x, red_y, green_x, green_y, image)
    return x, y, z, axs

def find_all_tfl_candidates(c_image: np.ndarray, **kwargs):
    """
     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
     :param kwargs: Whatever config you want to pass in here
     :return: 4-tuple of x_red, y_red, x_green, y_green
     """
    green_i_list, green_j_list = [], []
    red_i_list, red_j_list = [], []
    actual_image = c_image
    # c_image = c_image / 255
    c_image = c_image.astype(float)
    c_image = c_image[:, :, [0]]
    kernel = np.array([[0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    kernel = np.resize(kernel, (5, 5, 1))
    kernel = kernel.astype(float)
    # kernel = np.fliplr(kernel)
    # kernel = np.flipud(kernel)
    kernel = kernel - np.average(kernel)
    t = sg.convolve(c_image, kernel, mode='same')
    filterd_image = ndimage.maximum_filter(t, size=250)
    filterd_image = filterd_image == t
    green_j, green_i, z = np.where(filterd_image)
    for i in range(len(z)):
        green_pixel_color = actual_image[green_j[i], green_i[i], [1]]
        red_pixel_color = actual_image[green_j[i], green_i[i], [0]]
        if green_pixel_color > red_pixel_color:
            green_i_list.append(green_i[i])
            green_j_list.append(green_j[i])
        else:
            red_i_list.append(green_i[i])
            red_j_list.append(green_j[i])
    return red_i_list, red_j_list, green_i_list, green_j_list

def verify_candidate_images(red_x, red_y, green_x, green_y , image):
    all_candidates = []
    new_x, new_y = [], []
    # if red tfl is positive
    i = 0
    while i < len(red_x):
        temp_img = image[red_y[i] - 40:red_y[i] + 41, red_x[i] - 40 : red_x[i] + 41 ]
        if temp_img.shape == (81, 81, 3):
            candidate = np.array([]).astype('uint8')
            candidate = np.append(candidate , temp_img).astype('uint8')
            all_candidates.append(candidate)
            new_x.append(red_x[i])
            new_y.append(red_y[i])
        i += 1
    i = 0
    while i < len(green_x):
        temp_img = image[green_y[i] - 40:green_y[i] + 41, green_x[i] - 40: green_x[i] + 41]
        if temp_img.shape == (81,81, 3):
           candidate = np.array([]).astype('uint8')
           candidate = np.append(candidate, temp_img).astype('uint8')
           all_candidates.append(candidate)
           new_x.append(green_x[i])
           new_y.append(green_y[i])
        i += 1
    return all_candidates, new_x, new_y

#Part 2
def printPrediction(image_path, x, y, axs):
    original = Image.open(image_path)
    actual_image = np.array(Image.open(image_path))
    axs[1].imshow(actual_image)
    green_x,green_y,red_x,red_y = [],[],[],[]
    for i in range(len(x)):
        green_pixel_color = actual_image[y[i], x[i], [1]]
        red_pixel_color = actual_image[y[i], x[i], [0]]
        if green_pixel_color > red_pixel_color:
            green_x.append(x[i])
            green_y.append(y[i])
        else:
            red_x.append(x[i])
            red_y.append(y[i])
    axs[1].plot(red_x, red_y, 'ro', color='r', markersize=4)
    axs[1].plot(green_x, green_y, 'ro', color='g', markersize=4)
    axs[1].set_ylabel('trafic lights')
    #plt.show()

def all_tfl_candidates_wo_plotting(image_path,image_name, json_path=None, fig_num=None):
    original=Image.open(image_path)
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    red_x, red_y, green_x, green_y = find_all_tfl_candidates(image, some_threshold=42)
    return verify_candidate_images(red_x,red_y,green_x,green_y,image)

if __name__ == '__main__':
    main()
