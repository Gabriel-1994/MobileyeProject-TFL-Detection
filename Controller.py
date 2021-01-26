import TFL_Man
from Calculations import CurrContainer, PrevContainer

#Read the Frames
with open('frames.pls') as f:
    read_data = f.readline()
    pkl_path = '0'
    frames = []
    for line in f:
        line = line.rstrip('\n')
        if pkl_path == '0':
            pkl_path = line
        else:
            frames.append(line)
#loop over each frame and send to TFL_MAN
for i in range(len(frames)):
    if i == 0:
        currframe = CurrContainer(frames[i])
        TFL_Man.work_on_frame(0,currframe,pkl_path)
    else:
        currframe = CurrContainer(frames[i])
        prevframe = PrevContainer(frames[i-1])
        TFL_Man.work_on_frame(prevframe, currframe,pkl_path)

