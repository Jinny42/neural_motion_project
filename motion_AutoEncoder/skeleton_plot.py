import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class Skeleton_plot:
    def __init__(self, data, joint_order):
        # Attaching 3D axis to the figure
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 30)
        self.ax.set_zlim(-100, 100)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title('3D Skeleton Sequence')

        # indexes of joints
        self.hips_idx = joint_order.index('Hips')
        self.rknee_idx = joint_order.index('RightKnee')
        self.rtoe_idx = joint_order.index('RightToe')
        self.chest_idx = joint_order.index('Chest2')
        self.head_idx = joint_order.index('Head')
        self.lshoulder_idx = joint_order.index('LeftShoulder')
        self.lhand_idx = joint_order.index('lhand')
        self.rshoulder_idx = joint_order.index('RightShoulder')
        self.rhand_idx =joint_order.index('rhand')
        self.lknee_idx = joint_order.index('LeftKnee')
        self.ltoe_idx =joint_order.index('LeftToe')

        # Get skeleton data(3D numpy array format)
        self.data = data
        self.lines = self.draw_initial_lines()



    def update_skeleton(self, frame):
        for line, data in zip(self.lines, self.draw_lines(frame)):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :])
            line.set_3d_properties(data[2, :])
            line.set_marker("o")
        return self.lines

    def draw_initial_lines(self):
        frame = self.data[0]

        hips = frame[self.hips_idx]
        rknee = frame[self.rknee_idx]
        rtoe = frame[self.rtoe_idx]
        chest = frame[self.chest_idx]
        head = frame[self.head_idx]
        lshoulder = frame[self.lshoulder_idx]
        lhand = frame[self.lhand_idx]
        rshoulder = frame[self.rshoulder_idx]
        rhand = frame[self.rhand_idx]
        lknee = frame[self.lknee_idx]
        ltoe = frame[self.ltoe_idx]

        lines = [self.ax.plot([hips[0], rknee[0]], [hips[1], rknee[1]], [hips[2], rknee[2]])[0],
                 self.ax.plot([hips[0], lknee[0]], [hips[1], lknee[1]], [hips[2], lknee[2]])[0],
                 self.ax.plot([hips[0], chest[0]], [hips[1], chest[1]], [hips[2], chest[2]])[0],
                 self.ax.plot([chest[0], rshoulder[0]], [chest[1], rshoulder[1]], [chest[2], rshoulder[2]])[0],
                 self.ax.plot([chest[0], lshoulder[0]], [chest[1], lshoulder[1]], [chest[2], lshoulder[2]])[0],
                 self.ax.plot([chest[0], head[0]], [chest[1], head[1]], [chest[2], head[2]])[0],
                 self.ax.plot([rshoulder[0], rhand[0]], [rshoulder[1], rhand[1]], [rshoulder[2], rhand[2]])[0],
                 self.ax.plot([lshoulder[0], lhand[0]], [lshoulder[1], lhand[1]], [lshoulder[2], lhand[2]])[0],
                 self.ax.plot([rknee[0], rtoe[0]], [rknee[1], rtoe[1]], [rknee[2], rtoe[2]])[0],
                 self.ax.plot([lknee[0], ltoe[0]], [lknee[1], ltoe[1]], [lknee[2], ltoe[2]])[0]]

        return lines

    def draw_lines(self, frame):
        lines = []
        frame = self.data[frame]

        hips = frame[self.hips_idx]
        rknee = frame[self.rknee_idx]
        rtoe = frame[self.rtoe_idx]
        chest = frame[self.chest_idx]
        head = frame[self.head_idx]
        lshoulder = frame[self.lshoulder_idx]
        lhand = frame[self.lhand_idx]
        rshoulder = frame[self.rshoulder_idx]
        rhand = frame[self.rhand_idx]
        lknee = frame[self.lknee_idx]
        ltoe = frame[self.ltoe_idx]

        lines = [np.array([[hips[0], rknee[0]], [hips[1], rknee[1]], [hips[2], rknee[2]]]),
        np.array([[hips[0], lknee[0]], [hips[1], lknee[1]], [hips[2], lknee[2]]]),
        np.array([[hips[0], chest[0]], [hips[1], chest[1]], [hips[2], chest[2]]]),
        np.array([[chest[0], rshoulder[0]], [chest[1], rshoulder[1]], [chest[2], rshoulder[2]]]),
        np.array([[chest[0], lshoulder[0]], [chest[1], lshoulder[1]], [chest[2], lshoulder[2]]]),
        np.array([[chest[0], head[0]], [chest[1], head[1]], [chest[2], head[2]]]),
        np.array([[rshoulder[0], rhand[0]], [rshoulder[1], rhand[1]], [rshoulder[2], rhand[2]]]),
        np.array([[lshoulder[0], lhand[0]], [lshoulder[1], lhand[1]], [lshoulder[2], lhand[2]]]),
        np.array([[rknee[0], rtoe[0]], [rknee[1], rtoe[1]], [rknee[2], rtoe[2]]]),
        np.array([[lknee[0], ltoe[0]], [lknee[1], ltoe[1]], [lknee[2], ltoe[2]]])]

        return lines


    def show(self):
        # Creating the Animation object
        line_ani = animation.FuncAnimation(self.fig, self.update_skeleton, len(self.data),
                                           interval=10, blit=False)
        # line_ani.save('visualize.gif')
        plt.show()

class Skeleton_inference_plot:
    def __init__(self, data, joint_order):
        # Attaching 3D axis to the figure
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        self.ax.set_zlim(0, 30)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title('3D Skeleton Sequence')

        # indexes of joints
        self.hips_idx = joint_order.index('Hips')
        self.rknee_idx = joint_order.index('RightKnee')
        self.rtoe_idx = joint_order.index('RightToe')
        self.chest_idx = joint_order.index('Chest2')
        self.head_idx = joint_order.index('Head')
        self.lshoulder_idx = joint_order.index('LeftShoulder')
        self.lhand_idx = joint_order.index('lhand')
        self.rshoulder_idx = joint_order.index('RightShoulder')
        self.rhand_idx =joint_order.index('rhand')
        self.lknee_idx = joint_order.index('LeftKnee')
        self.ltoe_idx =joint_order.index('LeftToe')
        self.inf_lkee_idx = joint_order.index('inf_LeftKnee')
        self.inf_ltoe_idx = joint_order.index('inf_LeftToe')
        self.inf_rkee_idx = joint_order.index('inf_RightKnee')
        self.inf_rtoe_idx = joint_order.index('inf_RightToe')

        # Get skeleton data(3D numpy array format)
        tmpy = np.copy(data[:, :, 1])
        tmpz = np.copy(data[:, :, 2])
        data[:, :, 2] = tmpy
        data[:, :, 1] = tmpz

        self.data = data
        self.lines = self.draw_initial_lines()



    def update_skeleton(self, frame):
        for line, data in zip(self.lines, self.draw_lines(frame)):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :])
            line.set_3d_properties(data[2, :])
            line.set_marker("o")
        return self.lines

    def draw_initial_lines(self):
        frame = self.data[0]

        hips = frame[self.hips_idx]
        rknee = frame[self.rknee_idx]
        rtoe = frame[self.rtoe_idx]
        chest = frame[self.chest_idx]
        head = frame[self.head_idx]
        lshoulder = frame[self.lshoulder_idx]
        lhand = frame[self.lhand_idx]
        rshoulder = frame[self.rshoulder_idx]
        rhand = frame[self.rhand_idx]
        lknee = frame[self.lknee_idx]
        ltoe = frame[self.ltoe_idx]
        inf_lknee = frame[self.inf_lkee_idx]
        inf_ltoe = frame[self.inf_ltoe_idx]
        inf_rknee = frame[self.inf_rkee_idx]
        inf_rtoe = frame[self.inf_rtoe_idx]

        lines = [
                 # self.ax.plot([hips[0], rknee[0]], [hips[1], rknee[1]], [hips[2], rknee[2]], color='b')[0],
                 # self.ax.plot([hips[0], lknee[0]], [hips[1], lknee[1]], [hips[2], lknee[2]], color='b')[0],
                 self.ax.plot([hips[0], chest[0]], [hips[1], chest[1]], [hips[2], chest[2]], color='b')[0],
                 self.ax.plot([chest[0], rshoulder[0]], [chest[1], rshoulder[1]], [chest[2], rshoulder[2]], color='b')[0],
                 self.ax.plot([chest[0], lshoulder[0]], [chest[1], lshoulder[1]], [chest[2], lshoulder[2]], color='b')[0],
                 self.ax.plot([chest[0], head[0]], [chest[1], head[1]], [chest[2], head[2]], color='b')[0],
                 self.ax.plot([rshoulder[0], rhand[0]], [rshoulder[1], rhand[1]], [rshoulder[2], rhand[2]], color='b')[0],
                 self.ax.plot([lshoulder[0], lhand[0]], [lshoulder[1], lhand[1]], [lshoulder[2], lhand[2]], color='b')[0],
                 # self.ax.plot([rknee[0], rtoe[0]], [rknee[1], rtoe[1]], [rknee[2], rtoe[2]], color='b')[0],
                 # self.ax.plot([lknee[0], ltoe[0]], [lknee[1], ltoe[1]], [lknee[2], ltoe[2]], color='b')[0],
                 self.ax.plot([hips[0], inf_lknee[0]], [hips[1], inf_lknee[1]], [hips[2], inf_lknee[2]], color='r')[0],
                 self.ax.plot([hips[0], inf_rknee[0]], [hips[1], inf_rknee[1]], [hips[2], inf_rknee[2]], color='g')[0],
                 self.ax.plot([inf_lknee[0], inf_ltoe[0]], [inf_lknee[1], inf_ltoe[1]], [inf_lknee[2], inf_ltoe[2]], color='r')[0],
                 self.ax.plot([inf_rknee[0], inf_rtoe[0]], [inf_rknee[1], inf_rtoe[1]], [inf_rknee[2], inf_rtoe[2]], color='g')[0]]

        return lines

    def draw_lines(self, frame):
        frame = self.data[frame]

        hips = frame[self.hips_idx]
        rknee = frame[self.rknee_idx]
        rtoe = frame[self.rtoe_idx]
        chest = frame[self.chest_idx]
        head = frame[self.head_idx]
        lshoulder = frame[self.lshoulder_idx]
        lhand = frame[self.lhand_idx]
        rshoulder = frame[self.rshoulder_idx]
        rhand = frame[self.rhand_idx]
        lknee = frame[self.lknee_idx]
        ltoe = frame[self.ltoe_idx]
        inf_lknee = frame[self.inf_lkee_idx]
        inf_ltoe = frame[self.inf_ltoe_idx]
        inf_rknee = frame[self.inf_rkee_idx]
        inf_rtoe = frame[self.inf_rtoe_idx]

        lines = [
        # np.array([[hips[0], rknee[0]], [hips[1], rknee[1]], [hips[2], rknee[2]]]),
        # np.array([[hips[0], lknee[0]], [hips[1], lknee[1]], [hips[2], lknee[2]]]),
        np.array([[hips[0], chest[0]], [hips[1], chest[1]], [hips[2], chest[2]]]),
        np.array([[chest[0], rshoulder[0]], [chest[1], rshoulder[1]], [chest[2], rshoulder[2]]]),
        np.array([[chest[0], lshoulder[0]], [chest[1], lshoulder[1]], [chest[2], lshoulder[2]]]),
        np.array([[chest[0], head[0]], [chest[1], head[1]], [chest[2], head[2]]]),
        np.array([[rshoulder[0], rhand[0]], [rshoulder[1], rhand[1]], [rshoulder[2], rhand[2]]]),
        np.array([[lshoulder[0], lhand[0]], [lshoulder[1], lhand[1]], [lshoulder[2], lhand[2]]]),
        # np.array([[rknee[0], rtoe[0]], [rknee[1], rtoe[1]], [rknee[2], rtoe[2]]]),
        # np.array([[lknee[0], ltoe[0]], [lknee[1], ltoe[1]], [lknee[2], ltoe[2]]]),
        np.array([[hips[0], inf_lknee[0]], [hips[1], inf_lknee[1]], [hips[2], inf_lknee[2]]]),
        np.array([[hips[0], inf_rknee[0]], [hips[1], inf_rknee[1]], [hips[2], inf_rknee[2]]]),
        np.array([[inf_lknee[0], inf_ltoe[0]], [inf_lknee[1], inf_ltoe[1]],[inf_lknee[2], inf_ltoe[2]]]),
        np.array([[inf_rknee[0], inf_rtoe[0]], [inf_rknee[1], inf_rtoe[1]],[inf_rknee[2], inf_rtoe[2]]])
        ]

        return lines


    def show(self, path = None):
        # Creating the Animation object
        line_ani = animation.FuncAnimation(self.fig, self.update_skeleton, len(self.data),
                                           interval=30, blit=False)
        if path is not None:
            line_ani.save(path)
        plt.show()

# joint_order = ('Hips', 'RightKnee', 'RightToe', 'Chest2', 'Head', 'LeftShoulder', 'lhand', 'RightShoulder',
#                       'rhand', 'LeftKnee', 'LeftToe')
# data = np.load('preprocessed_data/raw_position/01_02.npy')

# joint_order = ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
#                         'Hips', 'Chest2', 'Head',
#                         'RightKnee', 'RightToe', 'RightShoulder','rhand')
# data = np.load('preprocessed_data/ordered_position/02_10.npy')
# plotter = Skeleton_plot(data, joint_order)
# plotter.show()

# joint_order = ('lhand', 'LeftShoulder', 'LeftToe', 'LeftKnee',
#                         'Hips', 'Chest2', 'Head',
#                         'RightKnee', 'RightToe', 'RightShoulder','rhand',
#                'inf1_LeftToe','inf1_LeftKnee',
#                'inf2_LeftToe','inf2_LeftKnee',
#                'inf3_LeftToe','inf3_LeftKnee')
# data = np.load('preprocessed_data/clips/for_inference/143_21/position.npy')
# plotter = Skeleton_inference_plot(data, joint_order)
# plotter.show()