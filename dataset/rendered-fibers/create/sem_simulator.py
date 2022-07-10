"""Create many images and log them to wandb"""

from tqdm import tqdm
import subprocess
import wandb
from typing import Tuple
from PIL import Image
import json
import pandas as pd
import numpy as np

wandb.login()

def gen_table(d):
    """create a table version of the task"""
    for k,v in d.items():
        v.update({"index":k})
        yield v


def create_rendered_dataset(sizes:Tuple[int, int, int], **kwargs):
    """Creates a simple dataset and logs it to wandb.

    Args:
        sizes: Sizes of the train val test splits
    """
    with wandb.init(project='diameterY', job_type='create-data', mode='online') as run:
        raw_data = wandb.Artifact(
            'rendered-fibers-mini', type='dataset',
            description='Single straight fibers split into train/val/test',
            metadata={
                'train_val_test':sizes,
            }
        )

        for set_name, set_size in zip(['train', 'val', 'test'], sizes):
            for i in tqdm(range(set_size), desc=set_name):
                task_uid = f'{set_name}{i:04d}'
                succeed = 0
                while succeed == 0:
                    try:
                        cmd = f'docker run --rm -i -u $(id -u):$(id -g) --volume "$(pwd):/kubric" kubruntu_sdf /usr/bin/python3 "create/sem_worker.py" {task_uid}'
                        subprocess.run(cmd, shell=True, check=True)
                        succeed=1
                    except Exception as e:
                        print(e)
                # --- load the image to log it to WandB
                im = Image.open(f'output/{task_uid}.png')
                seg = np.load(f'output/{task_uid}_seg.npz')['y']
                # with open(f'output/{task_uid}.json') as file:
                #     task_data = json.load(file)
                # table = wandb.Table(data=pd.DataFrame(gen_table(task_data)))
                # wandb_im = wandb.Image(im, caption=task_uid, masks={
                #     'ground_truth':{
                #         'mask_data':np.squeeze(seg),
                #         'class_labels':{int(x):f'f{int(x-100):02d}' for x in task_data.keys()}    
                #     }
                # })
                # raw_data.add(wandb_im, name=task_uid)
                # raw_data.add(table, name=f'{task_uid}_params')
                raw_data.add_file(f'output/{task_uid}.png', name=task_uid)
                raw_data.add_file(f'output/{task_uid}_seg.npz', name=f'{task_uid}_seg')
                raw_data.add_file(f'output/{task_uid}.json', name=f'{task_uid}_params')
        run.log_artifact(raw_data)
if __name__ == "__main__":
    create_rendered_dataset([256,64,64])

# for i in tqdm(range(10)):
    
#     task_uid = f'{i:04d}'
#     cmd = f'docker run --rm -i -u $(id -u):$(id -g) --volume "$(pwd):/kubric" kubruntu_sdf /usr/bin/python3 "create/sem_worker.py" {task_uid}'
#     subprocess.run(cmd, shell=True, check=True)

#     import numpy as np
#     seg = np.load(f'output/{task_uid}_seg.npz')['y']
