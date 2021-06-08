import dmc2gym
from PIL import Image
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--difficulty', type=str or None, default=None)
parser.add_argument('--davis_path', type=str, default="~/datasets/DAVIS/JPEGImages/480p/")
parser.add_argument('--output_dir', type=str, default="outputs")
parser.add_argument('--dynamic', action='store_true', default=False)
args = parser.parse_args()


env = dmc2gym.make(domain_name='reacher',
                   task_name='easy',
                   seed=1,
                   from_pixels=True,
                   visualize_reward=False,
                   height=84, width=84,
                   difficulty=args.difficulty,
                   background_dataset_path=os.path.expanduser(args.davis_path),
                   dynamic=args.dynamic,
                   )

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

done = False
obs = env.reset()
time_step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time_step += 1

    filepath = os.path.join(args.output_dir, f'{time_step:03d}.jpg')
    # import ipdb; ipdb.set_trace()
    image = Image.fromarray(np.swapaxes(obs, 0, 2))

    image.save(filepath)
