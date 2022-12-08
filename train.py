import os
import time
import argparse
import environment
import agent

from tqdm import tqdm
from init import *
from torch.utils.tensorboard import SummaryWriter


def plot_road(env: environment.Environment, save_dir: str):
    plt.clf()
    plt.cla()
    plt.axis('equal')
    plt.grid(0)
    plt.plot(*env.path_points.T, c="r")
    plt.plot(*env.road_polygon.T, c="b")
    plt.savefig(os.path.join(save_dir, "init.jpg"), dpi=200)
    plt.close("all")


def plot_result(file_dir, epoch, env: environment.Environment, car: agent.Agent, label="e"):
    hist = car.get_hist()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.axis('equal')
    ax1.plot(*env.road_polygon.T, c="b")
    ax1.plot(*hist.T[:2], "r--")

    ax2.plot(hist.T[-1], "b")

    plt.savefig(os.path.join(file_dir, f"{label}{epoch}.jpg"), dpi=200)
    plt.clf()
    plt.cla()
    plt.close("all")


def main(args):
    epochs, max_steps, eta, gamma, pretrained_table, recreate, seed, tot_length, path_width, length_range, dir_range = \
        list(vars(args).values())

    set_seed(seed)

    str_time = time.strftime("%Y%m%d_%H_%M", time.localtime(time.time()))
    save_dir = f"runs/{str_time}"
    os.makedirs(save_dir)

    tb_writer = SummaryWriter(log_dir=save_dir)

    car = agent.Agent(path_width=path_width)
    car.set_eta_gamma(eta=eta, gamma=gamma)

    if pretrained_table in os.listdir():
        print("Load pretrained table:", pretrained_table)
        car.load_q_tabel(pretrained_table)

    env = environment.Environment(tot_length=tot_length,
                                  path_width=path_width,
                                  length_range=length_range,
                                  dir_range=dir_range)

    # Train
    best_score = float("-inf")
    best_passed = 200

    if not recreate:
        plot_road(env, save_dir)

    for epoch in tqdm(range(epochs)):
        # print(f"Training epoch {epoch}...", end=" ")

        score = 0
        env.reset(car, recreate_road=recreate)

        # 初始化ob
        observation = env.observe(car)[1]
        radar_observation = observation.T[-1]
        in_road0, tot_passed, path_passed = env.completion(car.position)

        # 消除PyCharm警告用的。实际并没什么卵用，不加这句话程序也可以正常进行
        step, new_passed = 0, 0

        for step in range(max_steps):
            # 初始化本次奖励
            reward = 0

            # 获取初识速度
            v0 = car.velocity

            # 根据雷达ob和速度ob，小车做出下一步的动作
            action_made = car.step_next(
                observation=(radar_observation, v0),
                episode=epoch
            )

            # 更新ob
            observation_next = env.observe(car)[1]
            radar_observation_next = observation_next.T[-1]
            in_road, new_passed, new_path_passed = env.completion(car.position)

            # 如果出赛道、即将完成或者扣分太多，那么提前终止
            terminated = True if in_road == -1 else False
            if new_passed / tot_length >= 0.98 or score < -500000:
                terminated = True

            # Reward给奖励
            # 宝宝巴士加油就喂糖糖
            if car.acceleration > 0:
                reward += 4
            if car.acceleration > 0.5:
                reward += 5
            if car.acceleration < 0:
                reward -= 3
            if car.acceleration < 0.4:
                reward -= 5

            # 太靠边了要扣大分噢！！
            if radar_observation.min() / path_width <= 0.1:
                reward -= 50

            # 高速喂糖，低速惩罚，0 速度扣大分！
            reward += ((car.velocity // 3) - 1) * 3
            if car.velocity <= 1:
                reward -= 30

            # 进度++ 喂糖，成功进入下一个弯道给大大滴好处，往回走或者出赛道，扣大大滴分
            if new_passed > tot_passed:
                reward += 1
            elif new_passed < tot_passed:
                reward -= 10
            if in_road - in_road0 == 1:
                reward += 200
            elif in_road < in_road0:
                reward -= 20

            # 如果一直在第一段路，那么扣分也是多多滴，请加速走出第一段
            reward -= (2 if in_road < 1 else 0)

            # 每多存活150步，就加一些分数，别死太早！
            if step % 150 == 0 and step < 600:
                reward += step // 5

            # 出赛道扣大分！
            if in_road < 0 and new_passed / tot_length < 0.95:
                reward -= 1000

            # 完成比赛加分！
            if new_passed / tot_length >= 0.98:
                reward += 1000

            # 奖励累计，作为本次epoch的分数
            score += reward

            # 更新Q—Table：当前状态observation-采取某个动作action-得到回报reward-进入一个新的状态ob
            car.update_Q_table((radar_observation, v0), action_made, reward, (radar_observation_next, car.velocity))

            # 已经结束咧！！！
            if terminated:
                break

            # 刷新当前ob为新ob
            # observation = observation_next
            radar_observation = radar_observation_next
            in_road0, tot_passed, path_passed = in_road, new_passed, new_path_passed

        # print(f"Done! Steps: {step:04d}, Score: {score}")

        # 每500 epoch保存一次：
        if epoch % 500 == 0:
            plot_result(save_dir, epoch, env, car)
            car.save_q_tabel("last")
        # 完成的保存一次
        # if tot_passed / tot_length >= 0.97:
        #     plot_result(save_dir, epoch, env, car, label="finished")
        # 新的最高分保存一次
        if score > best_score and tot_passed > best_passed:
            plot_result(save_dir, epoch, env, car, label="better")
            best_passed = tot_passed
            best_score = score

        tb_writer.add_scalar("Passed", new_passed, epoch)
        tb_writer.add_scalar("Score", score, epoch)
        tb_writer.add_scalar("Step", step, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=agent.ETA)
    parser.add_argument('--gamma', type=float, default=agent.GAMMA)
    parser.add_argument('--load_table', type=str, default="last.npy")
    parser.add_argument('--recreate', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=968753241)
    parser.add_argument('--tot_length', type=int, default=5000)
    parser.add_argument('--path_width', type=int, default=environment.PATH_WIDTH)
    parser.add_argument('--length_range', type=tuple, default=(600, 800))
    parser.add_argument('--dir_range', type=tuple, default=(np.pi / 6, np.pi / 3))

    opt = parser.parse_args()

    main(opt)
