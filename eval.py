import init
import environment
import agent
import time
from utils import *
from matplotlib.animation import FuncAnimation

t0 = time.time()

init.set_seed(random.randint(0, 1 << 16))

path_width = environment.PATH_WIDTH
colors = ["red", "orangered", "darkorange", "orange", "gold", "yellow", "greenyellow", "limegreen", "green"]

env = environment.Environment(5000)
car = agent.Agent()
car.load_q_tabel("last.npy")

radar_directions = theta_e(np.array(agent.RADAR_DIRS).reshape(-1, 1) + np.pi / 2)

fig = plt.figure(figsize=(9, 6), dpi=200)
spec = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[5, 2], height_ratios=[1, 1, 1])
ax0 = fig.add_subplot(spec[:2, :1])
ax1 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, 1])
ax3 = fig.add_subplot(spec[2, :])
ax0.axis("equal")
ax0.axes.xaxis.set_major_locator(plt.NullLocator())
ax0.axes.yaxis.set_major_locator(plt.NullLocator())
ax0.grid(0)
ax1.axis("equal")
ax1.axes.xaxis.set_major_locator(plt.NullLocator())
ax1.axes.yaxis.set_major_locator(plt.NullLocator())
ax1.set_ylim(-50, +path_width * 2)
ax1.grid(0)
ax2.axis("equal")
ax2.axes.xaxis.set_major_locator(plt.NullLocator())
ax2.axes.yaxis.set_major_locator(plt.NullLocator())
ax2.set_xlim(-1, car.num_r_actions)
ax2.set_ylim(-1, car.num_acc_actions)
ax2.grid(0)
ax3.grid(0)


def refresh_board(envi: environment.Environment, agent_: agent.Agent, obs, dirs, action_hist, i):
    history = agent_.get_hist()[:i + 1]
    ax0.cla()
    ax0.plot(*envi.road_polygon.T, "b")
    ax0.quiver(*history[-1][:4])
    ax0.plot(*history.T[:2], "r--")
    ax0.scatter(*obs[i].T[:2], c="r")

    ax1.cla()
    ax1.scatter(*(dirs * path_width * 2).T, c="g", s=8)
    ax1.scatter(0, 0, c="b", s=10)
    color_digitized = np.digitize(obs[i].T[-1] / path_width, agent_.radar_bins)
    for j, r in enumerate(obs[i].T[-1]):
        x, y = r * dirs[j]
        ax1.plot([0, x], [0, y], color=colors[color_digitized[j]])

    ax2.cla()
    acc, r = action_hist[i] % agent_.num_acc_actions, action_hist[i] // agent_.num_acc_actions
    ax2.scatter(r, acc, c="r", s=100)

    ax3.cla()
    ax3.plot(agent_.hist_v[:i + 1])


def main():
    env.reset(car, recreate_road=False)
    in_road, observation = env.observe(car)
    observations = [observation]
    actions = []
    while in_road != -1:
        radar_observation = observation.T[-1]

        action_made = car.step_next(
            observation=(radar_observation, car.velocity),
            training=False
        )

        in_road, passed, _ = env.completion(car.position)
        if passed / env.tot_length > 0.95:
            in_road = -1
        observation = env.observe(car)[1]

        observations.append(observation)
        actions.append(action_made)

    def update(i):
        refresh_board(env, car, observations, radar_directions, actions, i)

    ani = FuncAnimation(fig, update, frames=len(actions), interval=33)
    ani.save("ani.gif")
    plt.show()


if __name__ == "__main__":
    main()
    print(f"Time Cost: {time.time() - t0 : 0.3f}")
