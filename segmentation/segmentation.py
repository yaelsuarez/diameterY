"""python segmentation.py --seed=0 --epochs=1 --batch-size=8 --learning-rate=0.0001 --loss='binary_crossentropy' --dropout=0.3 --l0-kernel-size=3"""
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import keras
import keras.layers as layers
import wandb
from argparse import ArgumentParser

import pandas as pd
import sys

assert len(tf.config.list_physical_devices()) == 2

def parse_args(raw_args):
    parser = ArgumentParser(description='train a U-net from scratch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--l0_kernel_size', type=int)
    parser.add_argument('--initial_features', type=int)
    args = parser.parse_args(raw_args)
    return args

def load_task(dataset_path, task_id):
    # Load an image
    im = Image.open(os.path.join(dataset_path, task_id + '.png'))
    im = np.array(ImageOps.grayscale(im))
    im = im.reshape(im.shape+(1,)).astype(np.float32)/255 # one channel image
    # Load segmentation
    seg:np.ndarray = np.load(os.path.join(dataset_path, task_id+ '_seg.npz'))['y']
    return im, seg

def select_point_and_fiber(seg):
    # Select a random point that is not background, return the mask for the fiber that the point touches.
    mask_all = seg > 0
    possible_points = np.argwhere(mask_all)
    point_index = np.random.randint(0, possible_points.shape[0]-1)
    point = possible_points[point_index]
    fiber_id = seg[point[0], point[1], point[2]]
    mask = seg == fiber_id
    selected_seg = np.zeros_like(seg, dtype=np.float32)
    selected_seg[mask] = 1.0
    return point[0:2], selected_seg

def get_example(dataset_path, task_id):
    """Creates an example for training"""
    im, seg = load_task(dataset_path, task_id)
    point, selected_seg = select_point_and_fiber(seg)
    point_channel = np.zeros_like(im, dtype=np.float32)
    point_channel[point[0], point[1], 0] = 1.0
    x = np.concatenate([im, point_channel], axis=-1)
    y = selected_seg
    return x, y

# generators for model training input
def example_generator(dataset_path, task_list):
    for task_id in task_list:
        yield get_example(dataset_path, task_id)

def batcher(generator, batch_size):
    batch = [], []
    counter = 0
    for x, y in generator:
        batch[0].append(x)
        batch[1].append(y)
        counter += 1
        if counter % batch_size == 0:
            yield batch
            batch = [],[]
    if len(batch[0]) > 0:
        yield batch

def train_batch_gen():
    train_tasks = [f"train{i:04d}" for i in range(256)]
    for x, y in batcher(example_generator(dataset_path, train_tasks), batch_size):
        yield x, y

def val_batch_gen():
    val_tasks = [f"val{i:04d}" for i in range(64)]
    for x, y in batcher(example_generator(dataset_path, val_tasks), batch_size):
        yield x, y

def get_unet(input_size, dropout, l0_kernel_size, initial_features, num_classes=1):
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(initial_features, l0_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(inputs)
    conv1 = layers.Conv2D(initial_features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(initial_features*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool1)
    conv2 = layers.Conv2D(initial_features*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(initial_features*2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool2)
    conv3 = layers.Conv2D(initial_features*2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(initial_features*2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool3)
    conv4 = layers.Conv2D(initial_features*2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv4)
    drop4 = layers.Dropout(dropout)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(initial_features*2**4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(pool4)
    conv5 = layers.Conv2D(initial_features*2**4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv5)
    drop5 = layers.Dropout(dropout)(conv5)

    up6 = layers.Conv2D(initial_features*2**3, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.concat([drop4,up6], axis = 3)
    conv6 = layers.Conv2D(initial_features*2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge6)
    conv6 = layers.Conv2D(initial_features*2**3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv6)

    up7 = layers.Conv2D(initial_features*2**2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.concat([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(initial_features*2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge7)
    conv7 = layers.Conv2D(initial_features*2**2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv7)

    up8 = layers.Conv2D(initial_features*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.concat([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(initial_features*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge8)
    conv8 = layers.Conv2D(initial_features*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv8)

    up9 = layers.Conv2D(initial_features, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = tf.concat([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(initial_features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(merge9)
    conv9 = layers.Conv2D(initial_features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv9)
    conv10 = layers.Conv2D(num_classes, 1, activation = 'sigmoid', padding='same')(conv9)

    model = keras.Model(inputs, conv10)
    return model

def log_test_table(test_batch_path:str, model):
    test_batch = np.load(test_batch_path)
    pred = model.predict(test_batch['x'])
    print(pred.shape)
    rows = []
    for i, (x, y, ŷ ) in enumerate(zip(test_batch['x'], test_batch['y'], pred)):
        im = wandb.Image(x[:,:,0], caption=f"test_batch_0_{i}",
            masks={
                "ground_truth":{
                    'mask_data':y[:,:,0].astype(int),
                    'class_labels':{0:'background', 1:'foreground'}
                },
                "selection":{
                    'mask_data':x[:,:,1].astype(int),
                    'class_labels':{0:'background', 1:'foreground'}
                },
                "prediction":{
                    'mask_data':(ŷ[:,:,0]>0.5).astype(int),
                    'class_labels':{0:'background', 1:'foreground'}
                },
            }
        )
        rows.append({'image':im})
    df = pd.DataFrame(rows)
    table = wandb.Table(dataframe=df)
    wandb.log({"test_batch_0_table":table})

if __name__ == '__main__':
    run = wandb.init(config=parse_args(sys.argv[1:]))
    c = run.config

    tf.keras.utils.set_random_seed(
        c.seed
    )

    dataset_path = '/home/fer/projects/diameterY/dataset/rendered-fibers/output'
    batch_size = c.batch_size

    train_dataset:tf.data.Dataset = tf.data.Dataset.from_generator(
        train_batch_gen, output_signature=(
            tf.TensorSpec(shape=[batch_size,256,256,2], dtype=tf.float32),
            tf.TensorSpec(shape=[batch_size,256,256,1], dtype=tf.float32))
    ).prefetch(2)

    val_dataset = tf.data.Dataset.from_generator(
        val_batch_gen, output_signature=(
            tf.TensorSpec(shape=[batch_size,256,256,2], dtype=tf.float32),
            tf.TensorSpec(shape=[batch_size,256,256,1], dtype=tf.float32))
    ).prefetch(2)

    unet = get_unet(
        [None,None,2], 
        c.dropout, 
        c.l0_kernel_size, 
        c.initial_features)
    
    adam = tf.keras.optimizers.Adam(learning_rate=c.learning_rate)
    precision=tf.keras.metrics.Precision()
    recall=tf.keras.metrics.Recall()
    iou = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
    
    unet.compile(
        optimizer=adam,
        loss=c.loss,
        metrics=[precision, recall, iou],
    )

    unet.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=c.epochs, 
        callbacks=[wandb.keras.WandbCallback()], 
        verbose=2
    )

    test_batch_path = '/home/fer/projects/diameterY/segmentation/test_batch_0.npz'
    log_test_table(test_batch_path, unet)

    task_id = 'train0001'
    im, seg = load_task(dataset_path, task_id)
