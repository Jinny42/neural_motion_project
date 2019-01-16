from pymo.features import *
from pymo.parsers import BVHParser
from pymo.preprocessing import *

def get_ED_error():
    path = 'result_data/exp1/inference_loss.txt'
    frame_list = []
    ED_list = []

    with open(path, 'r') as rf:
        lines = rf.readlines()

    for line in lines:
        frame_idx_from = line.find('[') + 1
        frame_idx_to = line.find(']')
        ED_idx_from = line.find('-') + 2
        ED_idx_to = line.find('\n')

        if frame_idx_from == -1:
            continue

        frame = line[frame_idx_from:frame_idx_to]
        ED = line[ED_idx_from:ED_idx_to]

        frame_list.append(eval(frame))
        ED_list.append(eval(ED))

    return frame_list, ED_list

def get_foot_heights():
    Rfoot_height_list = []
    Lfoot_height_list = []

    parser = BVHParser()

    parsed_data = parser.parse('motion_data/69/69_01.bvh')

    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])

    length = len(positions[0].values['RightToe_Yposition'])

    for i in range(length):
        Rfoot_height = positions[0].values['RightToe_Yposition'][i] /4
        Lfoot_height = positions[0].values['LeftToe_Yposition'][i] /4

        Rfoot_height_list.append(Rfoot_height)
        Lfoot_height_list.append(Lfoot_height)

    return Rfoot_height_list, Lfoot_height_list



frames, ED = get_ED_error()
RFH, LFH = get_foot_heights()

plt.plot(frames, ED, label='ED')
plt.plot(frames, RFH, label='Height of RightFoot/4')
plt.plot(frames, LFH, label='Height of LeftFoot/4')
plt.tight_layout()
plt.legend()
plt.grid(False)
plt.savefig('ED_foot_height.png')
plt.show()
