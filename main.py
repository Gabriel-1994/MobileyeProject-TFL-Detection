import DetectTFL
import Controller
#import SFM_standAlone
import tensorflow as tf
from keras.models import load_model

model = load_model('model.h5')


if __name__ == '__main__':
    for frame in Controller.frames:
        cand = DetectTFL.test_find_tfl_lights(frame, "image_name")
        for c in cand:
            c = c.reshape(1, 81, 81, 3)
            print(model.predict(c[0]))


