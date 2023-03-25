from models.model import Model as AutoLink
import gradio as gr
import PIL
import torch
import os
import imageio
import numpy as np
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default="celeba_wild_k32_m0.8_b16_t0.00075_sklr512")
args = parser.parse_args()


autolink = AutoLink.load_from_checkpoint(os.path.join("checkpoints", args.log, "model.ckpt"))
autolink.to(device)


def predict_image(image_in_img: PIL.Image.Image, image_in_video: PIL.Image.Image) -> PIL.Image.Image:
    if image_in_video == None and image_in_img == None:
        raise gr.Error("Please upload a video or image.")
    image_in = image_in_img if image_in_img else image_in_video
    edge_map = autolink(image_in)
    return edge_map


def predict_video(video_in: str) -> str:
    if video_in == None:
        raise gr.Error("Please upload a video or image.")
    video_out = video_in[:-4] + '_out.mp4'
    video_in = imageio.get_reader(video_in)
    writer = imageio.get_writer(video_out, mode='I', fps=video_in.get_meta_data()['fps'])
    for image_in in video_in:
        image_in = PIL.Image.fromarray(image_in)
        edge_map = autolink(image_in)
        writer.append_data(np.array(edge_map))
    writer.close()
    return video_out


def toggle(choice):
    if choice == "webcam":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


with gr.Blocks() as blocks:
    gr.Markdown("""
    # AutoLink
    ## Self-supervised Learning of Human Skeletons and Object Outlines by Linking Keypoints
    * [Paper](https://arxiv.org/abs/2205.10636)
    * [Project Page](https://xingzhehe.github.io/autolink/)
    * [GitHub](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints)
""")
    with gr.Tab("Video") as tab:
        with gr.Row():
            with gr.Column():
                video_or_file_opt = gr.Radio(["upload", "webcam"], value="upload",
                                             label="How would you like to upload your video?")
                video_in = gr.Video(source="upload", type="mp4")
                video_or_file_opt.change(fn=lambda s: gr.update(source=s, value=None), inputs=video_or_file_opt,
                                         outputs=video_in, queue=False)
            with gr.Column():
                video_out = gr.Video()
        run_btn = gr.Button("Run")
        run_btn.click(fn=predict_video, inputs=[video_in], outputs=[video_out])
        gr.Examples(fn=predict_video, examples=[["assets/00344.mp4"],],
                    inputs=[video_in], outputs=[video_out],
                    cache_examples=False)

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_or_file_opt = gr.Radio(["file", "webcam"], value="file",
                                             label="How would you like to upload your image?")
                image_in_video = gr.Image(source="webcam", type="pil", visible=False)
                image_in_img = gr.Image(source="upload",  type="pil", visible=True)

                image_or_file_opt.change(fn=toggle, inputs=[image_or_file_opt],
                                         outputs=[image_in_video, image_in_img], queue=False)
            with gr.Column():
                image_out = gr.Image()
        run_btn = gr.Button("Run")
        run_btn.click(fn=predict_image,
                      inputs=[image_in_img, image_in_video], outputs=[image_out])
        gr.Examples(fn=predict_image, examples=[["assets/jackie_chan.jpg", None]],
                    inputs=[image_in_img, image_in_video], outputs=[image_out],
                    cache_examples=False)

blocks.launch()

