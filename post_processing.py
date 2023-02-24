import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm

COLOURS = {
    'red': [200, 50, 50],
    'green': [90, 165, 90],
    'blue': [0, 0, 255],
    'purple': [112, 39, 195],
    'grey': [150, 150, 150],
    'white': [255, 255, 255],
    'black': [0, 0, 0],
    'yellow': [255, 255, 0]
}


def plot_video(
        agent, ball, target, action, action_title, reward, grid_size, video_name="video.mp4"
):
    '''
    agent: (n_step, n_agent, pos_dim=2)
    ball: (n_step, pos_dim=2)
    target: (n_step, pos_dim=2)
    '''
    height, width = grid_size
    from matplotlib import pyplot as plt
    import matplotlib.animation as manimation
    video_fps = 20
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=video_fps, metadata=metadata)
    fig = plt.figure(figsize=(width, height))
    total_steps = len(agent)
    with writer.saving(fig, video_name, dpi=300):
        for time in tqdm(range(total_steps)):
            fig.clf()
            plt.subplots_adjust(top=0.92, bottom=0.01, right=1, left=0, hspace=0, wspace=0)
            ax = fig.add_subplot(1, 1, 1)
            img = initialise_grid(ax, height, width)
            for i_agent, agent_pos in enumerate(agent[time]):
                ax.scatter(agent_pos[1], agent_pos[0], color="C00", s=2000, marker='o')
                ax.text(agent_pos[1] - 0.15, agent_pos[0] + 0.1, '%d' % (i_agent + 1), dict(size=30), color="C06")

            fill_list = [True, False, True]
            linestyle_list = ['-', '--', '-']
            for i, obj_pos in enumerate([ball[time], target[time]]):  # ball, target

                shape = plt.Circle([obj_pos[1], obj_pos[0]], 0.25, color="C0%d" % (i + 1),
                                   fill=fill_list[i], linestyle=linestyle_list[i])
                ax.add_patch(shape)
            # ax.set_xlim([-1,height])
            # ax.set_ylim([-1,width])
            plt.imshow(img, origin="upper")
            action_list = [action_title[a] for a in action[time]]
            plt.title("time %d" % time + " action " + ', '.join(action_list) + " reward " + str(reward[time]))
            # + " agent " + str( agent_pos) + " ball " + str(ball[time]) + " target " + str(target[time]))
            writer.grab_frame()
    plt.close(fig)


""" Initialise a gridworld grid """


def initialise_grid(ax, height, width):
    # TODO: can use this to plot final goal in the future
    # Initialise the map to all white
    img = [[COLOURS['white'] for _ in range(width)] for _ in range(height)]

    # Render the grid
    # for y in range(0, height):
    #     for x in range(0, width):
    #         if (x, y) == self.task.target_center:
    #             img[y][x] = COLOURS['red'] if self.goal_states[(x, y)] < 0 else COLOURS['green']
    #         elif (x, y) in self.agent_simulator.agents:
    #             img[y][x] = COLOURS['green']
    #         elif (x, y) in self.agent_simulator.agents:
    #             img[y][x] = COLOURS['blue']

    ax.xaxis.set_ticklabels([])  # clear x tick labels
    ax.axes.yaxis.set_ticklabels([])  # clear y tick labels
    ax.tick_params(which='both', top=False, left=False, right=False, bottom=False)
    ax.set_xticks([w - 0.5 for w in range(0, width, 1)])
    ax.set_yticks([h - 0.5 for h in range(0, height, 1)])
    ax.grid(color='lightgrey')
    return img

# def visualise_as_image(self, agent_position=None, title="", grid_size=1.5, gif=False):
#     # fig, ax, img = self.initialise_grid(grid_size=grid_size)
#     # current_position = (
#     #     self.get_initial_state() if agent_position is None else agent_position
#     # )
#
#     # Render the grid
#     for y in range(0, self.height):
#         for x in range(0, self.width):
#             if (x, y) == current_position:
#                 ax.scatter(x, y, s=2000, marker='o', edgecolors='none')
#             elif (x, y) in self.goal_states:
#                 plt.text(
#                     x,
#                     y,
#                     f"{self.get_goal_states()[(x, y)]:+0.2f}",
#                     fontsize="x-large",
#                     horizontalalignment="center",
#                     verticalalignment="center",
#                 )
#     im = plt.imshow(img, origin="lower")
#     plt.title(title)
#     if gif:
#         return fig, ax, im
#     else:
#
#
# return fig
