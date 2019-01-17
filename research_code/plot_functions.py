import matplotlib.pyplot as plt

def epoch_ED_plot(log_ED, show=False, save=False, path='result_data/Train_hist.png'):
    x = range(len(log_ED['train_ED']))

    y1 = log_ED['train_ED']
    y2 = log_ED['val_ED']

    plt.plot(x, y1, label='train_ED')
    plt.plot(x, y2, label='val_ED')

    plt.xlabel('Epoch')
    plt.ylabel('ED')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def frame_dist_plot_from_txt(path = 'result_data/inference_loss.txt') :
    frame_list = []
    ED_list = []

    with open(path, 'r') as rf:
        lines = rf.readlines()

    for line in lines:
        frame_idx_from = line.find('[') + 1
        frame_idx_to = line.find(']')
        ED_idx_from = line.find('-') + 2
        ED_idx_to = line.find('\n')

        if frame_idx_from == 0:
            continue

        frame = line[frame_idx_from:frame_idx_to]
        ED = line[ED_idx_from:ED_idx_to]

        frame_list.append(eval(frame))
        ED_list.append(eval(ED))

    plt.plot(frame_list, ED_list, label='ED')
    plt.xlabel('Frame')
    plt.ylabel('ED')
    plt.tight_layout()
    plt.legend()
    plt.grid(False)
    plt.savefig('Frame_ED.png')
    plt.show()


def draw_stickfigure3d(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8, 8)):
    #from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect(1)

    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints

    if data is None:
        df = mocap_track.values
    else:
        df = data

    for joint in joints_to_draw:
        parent_x = df['%s_Xposition' % joint][frame]
        parent_y = df['%s_Zposition' % joint][frame]
        parent_z = df['%s_Yposition' % joint][frame]
        # ^ In mocaps, Y is the up-right axis

        ax.scatter(xs=parent_x,
                   ys=parent_y,
                   zs=parent_z,
                   alpha=0.6, c='b', marker='o')

        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]

        for c in children_to_draw:
            child_x = df['%s_Xposition' % c][frame]
            child_y = df['%s_Zposition' % c][frame]
            child_z = df['%s_Yposition' % c][frame]
            # ^ In mocaps, Y is the up-right axis

            ax.plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 'k-', lw=2, c='black')

        if draw_names:
            ax.text(x=parent_x + 0.1,
                    y=parent_y + 0.1,
                    z=parent_z + 0.1,
                    s=joint,
                    color='rgba(0,0,0,0.9')

    return ax


# frame_dist_plot_from_txt(path = 'result_data/exp1/inference_loss.txt')
