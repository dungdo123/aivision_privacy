import os.path
import time

import gradio as gr
import cv2
import torch
import uuid
from pathlib import Path
from basicsr.utils.registry import ARCH_REGISTRY

from modules.face_swapper import *
from modules.face_restoration import *
import onnxruntime
from modules.lp_swapper import *
from ultralytics import YOLO
# from flask import Flask, request, session



class VIDEO_PRIVACY:
    def __init__(self):
        # Input params
        # self.webapp_manager = Flask(__name__)
        self.session_manager = list()
        self.SUBSAMPLE = 1
        self.lock_symbol = '\U0001F512'  # 🔒
        self.unlock_symbol = '\U0001F513'  # 🔓
        self.switch_values_symbol = '\U000021C5'  # ⇅
        self.test_stream_video_path = "E:/Projects/FACE/common_data/videos/01_dog.mp4"
        self.INPUT_FACE_STREAM_FILE = dict()
        self.OUTPUT_FACE_STREAM_FILE = dict()
        self.FIRST_FRAME_FACE_STREAM_FILE = dict()
        self.INPUT_LP_STREAM_FILE = dict()
        self.OUTPUT_LP_STREAM_FILE = dict()
        self.FIRST_FRAME_LP_STREAM_FILE = dict()

        self.SOURCE_FACE_PATH = "data/source_faces"
        self.SOURCE_FACE_CLASS = "data/source_face_classification"
        self.SWAP_FACE_MODEL = "models/swapface/aivision_swapface_v1.onnx"
        self.LP_SEGMENT_MODEL = "models/lp_detect/lp_detect.pt"
        self.FACE_RESTORE = False
        self.source_face_dict = dict()
        self.source_lp_dict = get_source_lp_dict()
        self.is_input_video_changed = 0

        # load machine default available providers
        self.providers = ['CUDAExecutionProvider','CPUExecutionProvider']# onnxruntime.get_available_providers()
        # print(f"list of providers{self.providers}")
        # load the face analyser
        self.face_analyser = get_face_analyser(self.providers)

        # Load LP segment model
        self.lp_segment_engine = YOLO(self.LP_SEGMENT_MODEL)

        # load face_swapper
        self.model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.SWAP_FACE_MODEL)
        self.face_swapper = get_face_swap_model(self.SWAP_FACE_MODEL)

        # Load model for face-restore: now support only ['codeformer']
        # if FACE_RESTORE:
        download_face_restore_check_ckpts()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        self.upsampler = set_realesrgan()
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                              codebook_size=1024,
                                                              n_head=8,
                                                              n_layers=9,
                                                              connect_list=["32", "64", "128", "256"],
                                                              ).to(self.device)
        self.codeformer_ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        self.codeformer_checkpoint = torch.load(self.codeformer_ckpt_path)["params_ema"]
        self.codeformer_net.load_state_dict(self.codeformer_checkpoint)
        self.codeformer_net.eval()

        # get source face list
        self.source_face_list = get_source_face_list(self.SOURCE_FACE_PATH, self.face_analyser)
        self.source_face_dict['random'] = self.source_face_list
        # Get source face classification list
        # sex: male/female || age: baby(0-3)/kid(3-15)/young(15-35)/middle(35-59)/old(60~)
        # male
        self.male_baby_sp = self.SOURCE_FACE_CLASS + "/male/baby"
        self.source_face_male_baby = get_source_face_list(self.male_baby_sp, self.face_analyser)
        self.source_face_dict['male_baby'] = self.source_face_male_baby
        self.male_kid_sp = self.SOURCE_FACE_CLASS + "/male/kid"
        self.source_face_male_kid = get_source_face_list(self.male_kid_sp, self.face_analyser)
        self.source_face_dict['male_kid'] = self.source_face_male_kid
        self.male_young_sp = self.SOURCE_FACE_CLASS + "/male/young"
        self.source_face_male_young = get_source_face_list(self.male_young_sp, self.face_analyser)
        self.source_face_dict['male_young'] = self.source_face_male_young
        self.male_middle_sp = self.SOURCE_FACE_CLASS + "/male/middle"
        self.source_face_male_middle = get_source_face_list(self.male_middle_sp, self.face_analyser)
        self.source_face_dict['male_middle'] = self.source_face_male_middle
        self.male_old_sp = self.SOURCE_FACE_CLASS + "/male/old"
        self.source_face_male_old = get_source_face_list(self.male_old_sp, self.face_analyser)
        self.source_face_dict['male_old'] = self.source_face_male_old
        # female
        self.female_baby_sp = self.SOURCE_FACE_CLASS + "/female/baby"
        self.source_face_female_baby = get_source_face_list(self.female_baby_sp, self.face_analyser)
        self.source_face_dict['female_baby'] = self.source_face_female_baby
        self.female_kid_sp = self.SOURCE_FACE_CLASS + "/female/kid"
        self.source_face_female_kid = get_source_face_list(self.female_kid_sp, self.face_analyser)
        self.source_face_dict['female_kid'] = self.source_face_female_kid
        self.female_young_sp = self.SOURCE_FACE_CLASS + "/female/young"
        self.source_face_female_young = get_source_face_list(self.female_young_sp, self.face_analyser)
        self.source_face_dict['female_young'] = self.source_face_female_young
        self.female_middle_sp = self.SOURCE_FACE_CLASS + "/female/middle"
        self.source_face_female_middle = get_source_face_list(self.female_middle_sp, self.face_analyser)
        self.source_face_dict['female_middle'] = self.source_face_female_middle
        self.female_old_sp = self.SOURCE_FACE_CLASS + "/female/old"
        self.source_face_female_old = get_source_face_list(self.female_old_sp, self.face_analyser)
        self.source_face_dict['female_old'] = self.source_face_female_old
        self.web_ui()

    def web_ui(self):
        with gr.Blocks() as self.demo:
            # FACE SWAP SLOT
            # with gr.Row():
            #     gr.set_static_paths(paths=["resource/icon/"])
            #     gr.HTML(
            #         """
            #         <img src="https://github.com/dungdo123/Code_Interview/blob/bb7c50389eac539105bab080c66b88c340afe843/resource_dev/aivisionin_icon.png">
            #         """
            #     )
            # session_handle = gr.State([])
            # print(session_handle)
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                gr.HTML(
                    """
                        <h1 style="text-align: center;font-size: 50px">
                         AIVISIONIN MVP DEMO - BATCH PROCESSING
                        </h1>
                        <h2 style="text-align: center;font-size: 30px">
                         De-Identification of Human Face
                        </h2>

                        """
                )
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                with gr.Column(scale=7):
                    input_display_1 = gr.Video(label="Input Video", streaming=True, autoplay=True)
                    upload_button_1 = gr.UploadButton(label="Upload Video")

                with gr.Column(scale=1, min_width=100):
                    run_button_1 = gr.Button("", icon='data/icon/swap_icon.png')

                with gr.Column(scale=7):
                    output_display_1 = gr.Video(label="Privacy Protected", streaming=True, autoplay=True,
                                                watermark="Protected")

            # VEHICLE LICENSE PLATE SLOT
            with gr.Row():
                gr.HTML(
                    """
                    """
                )

            with gr.Row():
                gr.HTML(
                    """
                        <h1 style="text-align: center;font-size: 30px">
                         De-Identification of Vehicle License Plate
                        </h1>

                        """
                )
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                with gr.Column(scale=7):
                    input_display_2 = gr.Video(label="Input Video", streaming=True, autoplay=True)
                    upload_button_2 = gr.UploadButton(label="Upload Video")

                with gr.Column(scale=1, min_width=100):
                    run_button_2 = gr.Button("", icon='data/icon/swap_icon.png')

                with gr.Column(scale=7):
                    output_display_2 = gr.Video(label="Privacy Protected", streaming=True, autoplay=True,
                                                watermark="Protected")

            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            # FULL BODY HIDE SLOT
            with gr.Row():
                gr.HTML(
                    """
                    """
                )

            with gr.Row():
                gr.HTML(
                    """
                        <h1 style="text-align: center;font-size: 30px">
                         De-Identification of Full Human Body
                        </h1>
                        <h1 style="text-align: center;font-size: 15px">
                         Coming Soon
                        </h1>

                        """
                )

            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            # upload and stream for face
            session_id = gr.State(get_session_id)
            upload_button_1.upload(self.upload_face_video, inputs=[upload_button_1,session_id], outputs=[input_display_1])
            run_button_1.click(fn=self.face_files_stream, inputs=[session_id], outputs=[input_display_1, output_display_1])
            upload_button_1.click(fn=self.check_session, inputs=[session_id])
            run_button_1.click(fn=self.check_session, inputs=[session_id])

            # upload and stream for lp
            upload_button_2.upload(self.upload_lp_video, inputs=[upload_button_2,session_id], outputs=[input_display_2])
            run_button_2.click(fn=self.lp_files_stream,inputs=[session_id], outputs=[input_display_2, output_display_2])
            upload_button_2.click(fn=self.check_session, inputs=[session_id])
            run_button_2.click(fn=self.check_session, inputs=[session_id])

    def launch(self, share=False):
        # allowed_paths = ["aivisionin_icon.png"]
        self.demo.launch(share=share)

    def check_session(self, session_id):
        self.session_manager.append(session_id)
        print(self.session_manager)

    def gen_face_swap_video(self, input_video_path,session_id, is_restore_face=False):
        # input video processing
        input_video_name = Path(input_video_path).name.split(".")[0]
        input_cap = cv2.VideoCapture(input_video_path)
        input_video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        input_fps = int(input_cap.get(cv2.CAP_PROP_FPS))
        input_desired_fps = input_fps // self.SUBSAMPLE
        output_desired_fps = input_desired_fps

        # input_batch_name = f"output/stream_output/input_{input_video_name}_{uuid.uuid4()}.mp4"
        output_batch_name = f"output/stream_output/output_{input_video_name}_{uuid.uuid4()}.mp4"

        # input_segment_file, output_segment_file = None, None
        output_segment_file = None
        source_target_map, frame_face_embeddings, input_batch_name, first_frame_name = get_unique_faces_from_target_video(input_video_path,
                                                                                            self.face_analyser,
                                                                                            self.source_face_dict)

        for frame in tqdm(frame_face_embeddings, desc=f"swap faces in each frame"):
            frame_location = frame['location']
            origin_frame = cv2.imread(frame_location)
            temp_frame = origin_frame.copy()

            output_height, output_width = temp_frame.shape[:2]
            if output_segment_file is None:
                # input_segment_file = cv2.VideoWriter(input_batch_name, input_video_codec, input_desired_fps,
                #                                      (output_width, output_height))
                output_segment_file = cv2.VideoWriter(output_batch_name, input_video_codec, output_desired_fps,
                                                      (output_width, output_height))
            face_swap_start = time.time()
            for i, target_face in enumerate(frame['faces']):
                face_id = target_face['target_centroid']
                source_face = source_target_map[face_id]['source']
                temp_frame = self.face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
            face_swap_end = time.time()
            # swap_processing_time = face_swap_end - face_swap_start
            # number_of_face = len(frame['faces'])
            # print("processing time with {} faces :{}".format(number_of_face, swap_processing_time))
            if is_restore_face:
                temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)
                temp_frame = face_restoration(temp_frame,
                                              background_enhance=True,
                                              face_upsample=True,
                                              upscale=1,
                                              codeformer_fidelity=0.5,
                                              upsampler=self.upsampler,
                                              codeformer_net=self.codeformer_net,
                                              device=self.device)

            output_segment_file.write(temp_frame)
            # input_segment_file.write(origin_frame)

        output_segment_file.release()
        # input_segment_file.release()

        self.INPUT_FACE_STREAM_FILE[session_id] = input_batch_name
        self.OUTPUT_FACE_STREAM_FILE[session_id] = output_batch_name
        self.FIRST_FRAME_FACE_STREAM_FILE[session_id] = first_frame_name

    def face_files_stream(self, session_id):
        if os.path.exists(self.INPUT_FACE_STREAM_FILE[session_id]) and os.path.exists(self.OUTPUT_FACE_STREAM_FILE[session_id]):
            yield self.INPUT_FACE_STREAM_FILE[session_id], self.OUTPUT_FACE_STREAM_FILE[session_id]

    def upload_face_video(self, filepath, session_id):
        # name = Path(filepath).name
        self.gen_face_swap_video(filepath,session_id)
        yield self.FIRST_FRAME_FACE_STREAM_FILE[session_id]

    def gen_lp_swap_video(self, filepath,session_id):
        self.INPUT_LP_STREAM_FILE[session_id], self.FIRST_FRAME_LP_STREAM_FILE[session_id], self.OUTPUT_LP_STREAM_FILE[session_id] = swap_lp_video_process_all(filepath, self.lp_segment_engine, self.source_lp_dict)

    def upload_lp_video(self, filepath, session_id):

        self.gen_lp_swap_video(filepath,session_id)
        yield  self.FIRST_FRAME_LP_STREAM_FILE[session_id]

    def lp_files_stream(self, session_id):

        if os.path.exists(self.INPUT_LP_STREAM_FILE[session_id]) and os.path.exists(self.OUTPUT_LP_STREAM_FILE[session_id]):
            yield self.INPUT_LP_STREAM_FILE[session_id], self.OUTPUT_LP_STREAM_FILE[session_id]


if __name__ == "__main__":
    webapp = VIDEO_PRIVACY()
    webapp.launch(share=False)

# evaluation - face
# 1 face size 1280x720: processing time: ~0.07 ~15fps
# 3 face size 1920x1080: processing time: ~0.4
# 1 face size x600 processing time: ~0.046
# 3 face size x600 processing time: ~0.14
# 1 face size [711, 400] processing time: ~0.027
# 3 face size [711, 400] processing time: ~0.082

