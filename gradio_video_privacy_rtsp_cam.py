import os

import gradio as gr
from torch.cuda import stream

from modules.face_swapper import *
from modules.lp_swapper import *
from ultralytics import YOLO

class VIDEO_PRIVACY:
    def __init__(self):
        # setting variables for each session
        self.session_manager = []
        self.stream_video_source = dict()
        self.stream_frame_size = dict()
        self.stream_fps = dict()
        self.selected_objects = dict()
        self.is_input_streaming = dict()
        self.streaming_length = dict()  # in number of frames, the time for running streaming, refresh the user browser to save memory
        self.vsource_type = dict()

        # static path to models and resource
        self.LP_SEGMENT_MODEL = "models/lp_detect/lp_detect.pt"


        # Loading model and resource
        self.lp_segment_engine = YOLO(self.LP_SEGMENT_MODEL)
        self.source_lp_dict = get_source_lp_dict()

        # static input parameters

        self.SUBSAMPLE = 1
        self.stream_bufer_length = 1800
        self.streaming_period = 3600
        # run webui
        self.web_ui()

    def web_ui(self):
        # with gr.Blocks(delete_cache=(60, 120)) as self.demo:
        with gr.Blocks(delete_cache=(60, 120)) as self.demo:
            with gr.Row():
                gr.HTML(
                    """
                    """
                )
            with gr.Row():
                gr.HTML(
                    """
                        <h1 style="text-align: center;font-size: 50px">
                         AIVISIONIN 
                        </h1>
                        <h2 style="text-align: center;font-size: 50px">
                         De-Identification MVP DEMO
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
                    input_display = gr.Video(label="Input Video", streaming=True, autoplay=False)
                    start_stream_btn = gr.Button("Start Streaming")
                    with gr.Accordion("Setting", open=False):

                        source = gr.Radio(choices=["Video", "RTSP", "Webcam"], label="Source")
                        with gr.Row():
                            source_tb = gr.Textbox(visible=False, label="Video Source", interactive=False)
                            source_btn = gr.UploadButton(label = "Upload",visible=False, interactive=False)
                        # rtsp_address = gr.Textbox(visible=True, label="RTSP Address", interactive=False)
                        source.change(self.source_update_visibility, inputs=[source], outputs=[source_tb, source_btn])

                        objects = gr.Radio(choices=["Human Face", "License Plate", "Face + LP"], label="Objects")
                        with gr.Row():
                            frame_size = gr.Number(value=600 ,visible=True, label="Frame Height", interactive=True)
                            stream_fps = gr.Number(value=30, visible=True, label="FPS", interactive=True)
                        apply_setting_btn = gr.Button("Apply")

                with gr.Column(scale=1, min_width=100):
                    swap_button = gr.Button("", icon='data/icon/swap_icon.png')

                with gr.Column(scale=7):
                    output_display = gr.Video(label="Privacy Protected", streaming=True, autoplay=False,
                                                watermark="Protected")
            # signals process
            session_id = gr.State(get_session_id)
            # self.is_input_streaming[session_id] = False
            # print(self.is_input_streaming[session_id])
            # print(session_id)
            apply_setting_btn.click(fn=self._setting_apply_btn, inputs=[session_id, source, source_btn,source_tb, objects, frame_size, stream_fps])
            start_stream_btn.click(fn=self.start_stop_streaming, inputs=[session_id], outputs=[start_stream_btn])
            start_stream_btn.click(fn=self.stream_input_video, inputs=[session_id], outputs=[input_display])
            swap_button.click(fn=self.stream_lp_swap_video, inputs=[session_id], outputs=[output_display])

            # restart streaming when it reached streaming_length
            input_display.end(fn=self.stream_input_video, inputs=[session_id], outputs=[input_display])
            output_display.end(fn=self.stream_lp_swap_video, inputs=[session_id], outputs=[output_display])



    def source_update_visibility(self, source):
        if source == "Video":
            return gr.Textbox(visible=False, label="RTSP Address", interactive=False), gr.UploadButton(label="Upload",visible=True, interactive=True)
        elif source == "RTSP":
            return gr.Textbox(visible=True, label="RTSP Address", interactive=True), gr.UploadButton(label="Upload",visible=False, interactive=False)
        else:  # Webcam
            return gr.Textbox(visible=False, label="RTSP Address", interactive=False), gr.UploadButton(label="Upload",visible=False, interactive=False)

    def start_stop_streaming(self, session_id):
        if self.is_input_streaming[session_id]:
            self.is_input_streaming[session_id] = False
            return gr.Button("Start Streaming")
        else:
            self.is_input_streaming[session_id] = True
            return gr.Button("Stop Streaming")

    def launch(self, share=False):
        self.demo.queue().launch(share=share)

    def _setting_apply_btn(self, session_id, source_option, video_source_txt,rtsp_addr_txt, selected_object, frame_size, fps):
        if session_id not in self.session_manager:
            self.session_manager.append(session_id)
        if session_id not in self.is_input_streaming.keys():
            self.is_input_streaming[session_id] = False
        if source_option == "Video":
            self.stream_video_source[session_id] = str(video_source_txt).replace(" ", "")
            self.selected_objects[session_id] = str(selected_object)
            self.stream_frame_size[session_id] = int(frame_size)
            self.stream_fps[session_id] = int(fps)
            self.streaming_length[session_id] = self.stream_fps[session_id]*self.streaming_period
            self.vsource_type[session_id] = source_option
            # return gr.Button("Start Streaming", interactive=False)
        elif source_option == "RTSP":
            self.stream_video_source[session_id] = str(rtsp_addr_txt)
            self.selected_objects[session_id] = str(selected_object)
            self.stream_frame_size[session_id] = int(frame_size)
            self.stream_fps[session_id] = int(fps)
            self.streaming_length[session_id] = self.stream_fps[session_id] * self.streaming_period
            self.vsource_type[session_id] = source_option
            # return gr.Button("Start Streaming", interactive=True)
        else:
            print("stream from webcam is developing")
            # return gr.Button("Start Streaming", interactive=False)

    def stream_input_video(self,session_id):
        print(self.session_manager)
        print(self.stream_video_source[session_id])
        if self.is_input_streaming[session_id]:
            input_cap = cv2.VideoCapture(self.stream_video_source[session_id])
            segment_writer = None
            frame_count = 0
            desired_fps = int(self.stream_fps[session_id])
            stream_batch_len = 2*desired_fps
            stream_buffer_mp4 = []
            stream_buffer_ts = []
            batch_frame_count = 0
            codec_writer = cv2.VideoWriter_fourcc(*"mp4v")
            segment_name = f"output/stream_output/intput_stream_{uuid.uuid4()}.mp4"

            while self.is_input_streaming[session_id] and frame_count < self.streaming_length[session_id]:
                ret, frame = input_cap.read()

                if ret:
                    original_height, original_width, _ = frame.shape

                    # stream fps

                    # Specify the desired height
                    desired_height = int(self.stream_frame_size[session_id])

                    # Calculate the aspect ratio
                    aspect_ratio = original_width / original_height

                    # Calculate the desired width based on the aspect ratio and desired height
                    desired_width = int(desired_height * aspect_ratio)

                    # Resize the image
                    resized_frame = cv2.resize(frame, (desired_width, desired_height))

                    if segment_writer is None:
                        segment_writer = cv2.VideoWriter(segment_name, codec_writer, desired_fps, (desired_width, desired_height))
                    if frame_count % self.SUBSAMPLE == 0:
                        batch_frame_count += 1
                        segment_writer.write(resized_frame)
                    if batch_frame_count == stream_batch_len:

                        batch_frame_count = 0
                        segment_writer.release()
                        stream_buffer_mp4.append(segment_name)
                        stream_buffer_ts.append(segment_name.replace(".mp4", ".ts"))

                        yield segment_name
                        if len(stream_buffer_mp4) == self.stream_bufer_length:
                            os.remove(stream_buffer_mp4[0])
                            os.remove(stream_buffer_ts[0])
                            stream_buffer_mp4.pop(0)
                            stream_buffer_ts.pop(0)
                        segment_name = f"output/stream_output/input_stream_{uuid.uuid4()}.mp4"
                        segment_writer = cv2.VideoWriter(segment_name, codec_writer, desired_fps,
                                                         (desired_width, desired_height))
                    frame_count += 1
                else:
                    break
            input_cap.release()
        # delete uploaded video
        if self.vsource_type[session_id] == "Video":
            os.remove(self.stream_video_source[session_id])

    def stream_lp_swap_video(self,session_id):
        if self.is_input_streaming[session_id]:
            input_cap = cv2.VideoCapture(self.stream_video_source[session_id])
            segment_writer = None
            frame_count = 0
            desired_fps = int(self.stream_fps[session_id])
            stream_batch_len = 2*desired_fps
            stream_buffer_mp4 = []
            stream_buffer_ts = []
            batch_frame_count = 0
            codec_writer = cv2.VideoWriter_fourcc(*"mp4v")
            segment_name = f"output/stream_output/intput_stream_{uuid.uuid4()}.mp4"

            while self.is_input_streaming[session_id] and frame_count < self.streaming_length[session_id]:
                ret, frame = input_cap.read()

                if ret:
                    original_height, original_width, _ = frame.shape

                    # stream fps

                    # Specify the desired height
                    desired_height = int(self.stream_frame_size[session_id])

                    # Calculate the aspect ratio
                    aspect_ratio = original_width / original_height

                    # Calculate the desired width based on the aspect ratio and desired height
                    desired_width = int(desired_height * aspect_ratio)

                    # Resize the image
                    resized_frame = cv2.resize(frame, (desired_width, desired_height))

                    if segment_writer is None:
                        segment_writer = cv2.VideoWriter(segment_name, codec_writer, desired_fps, (desired_width, desired_height))
                    if frame_count % self.SUBSAMPLE == 0:
                        batch_frame_count += 1
                        temp_frame = swap_lp_image(self.lp_segment_engine, resized_frame, self.source_lp_dict)
                        segment_writer.write(temp_frame)
                    if batch_frame_count == stream_batch_len:
                        batch_frame_count = 0
                        segment_writer.release()
                        stream_buffer_mp4.append(segment_name)
                        stream_buffer_ts.append(segment_name.replace(".mp4", ".ts"))

                        yield segment_name
                        if len(stream_buffer_mp4) == self.stream_bufer_length:
                            os.remove(stream_buffer_mp4[0])
                            os.remove(stream_buffer_ts[0])
                            stream_buffer_mp4.pop(0)
                            stream_buffer_ts.pop(0)
                        segment_name = f"output/stream_output/input_stream_{uuid.uuid4()}.mp4"
                        segment_writer = cv2.VideoWriter(segment_name, codec_writer, desired_fps,
                                                         (desired_width, desired_height))
                    frame_count += 1
                else:
                    break
            input_cap.release()


if __name__ == "__main__":
    webapp = VIDEO_PRIVACY()
    # black theme: /?__theme=dark
    webapp.launch(share=False)

# rtsp_url = "rtsp://172.19.192.1:554/live"