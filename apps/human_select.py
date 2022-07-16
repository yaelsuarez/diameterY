from tabnanny import verbose
from cv2 import threshold
import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
from itertools import islice
from PIL import Image, ImageDraw, ImageColor
from diametery.line_fit import LineFit
import random

MAX_SELECTIONS = 8

def load_model():
    return tf.keras.models.load_model('model-best.h5')

model = load_model()
colors = list(ImageColor.colormap.keys())
linefit = LineFit(30, 0.3)

def get_blob_centroids(mask):
    centers = []
    print(mask.dtype)
    print(mask.shape)
    contours, hierarchies = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append([cx,cy])
            print(cx,cy)
    return centers
    
def predict_mask(input_img, threshold):
    # unpack and reshape input
    im, mask = input_img["image"], input_img["mask"]
    mask = mask[:,:,0].astype(np.uint8)
    im = im.astype(np.float32)/256

    # get centroids to measure the fibers
    centers = get_blob_centroids(mask)

    # create a batch of input for the model
    batch = np.zeros([MAX_SELECTIONS,256,256,2], dtype=np.float32)

    for i, (cx, cy) in enumerate(islice(centers, MAX_SELECTIONS)):
        batch[i,:,:,0] = im
        batch[i,cy,cx,1] = 1.0
    
    pred = model.predict(batch, verbose=0).squeeze()
    # create a single image with the background and the foreground
    im = Image.fromarray(im*255).convert("RGBA")
    # m = Image.fromarray(pred[0]>threshold).convert("RGBA")
    # im = Image.blend(im, m, 0.5)
    imgd = ImageDraw.Draw(im)
    ds = []
    for p in islice(pred, len(centers)):
        d, lines = linefit.predict((p>threshold).astype(np.uint8)*255)
        ds.append(d)
        m = Image.fromarray(p>threshold)
        imgd.bitmap([0,0], m, fill=random.choice(colors))
        for line in lines:
            imgd.line(line, fill ="blue", width = 0)
        
    return im, ds

demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Column():
            img = gr.Image(
                tool="sketch", 
                source="upload",
                label="Mask",
                image_mode='L',
                shape=[256,256],
                value='/home/fer/projects/diameterY/dataset/real-fibers/val_image.jpeg'
            )
            threshold = gr.Slider(
                label='Segmentation threshold', minimum=0, maximum=1, value=0.5)

            with gr.Row():
                btn = gr.Button("Run")
        with gr.Column():
            img2 = gr.Image()
            text = gr.Text()

    btn.click(fn=predict_mask, inputs=[img, threshold], outputs=[img2,text])


demo.launch()