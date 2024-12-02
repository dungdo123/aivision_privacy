# Import libraries
# Torch==2.0.1+cu118
import gradio as gr
from numpy import random as np_rd

from basicsr.utils.registry import ARCH_REGISTRY

from modules.face_swapper import *
from modules.face_restoration import *
import onnxruntime
import copy


# Input setting
# input params setting
SOURCE_FACE_PATH = "data/source_faces"
SOURCE_FACE_CLASS = "data/source_face_classification"
SWAP_FACE_MODEL = "models/swapface/aivision_swapface_v1.onnx"
FACE_RESTORE = False
source_face_dict = dict()

# load machine default available providers
providers = onnxruntime.get_available_providers()
print(f"list of providers{providers}")
# load the face analyser
face_analyser = get_face_analyser(providers)

# load face_swapper
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), SWAP_FACE_MODEL)
face_swapper = get_face_swap_model(SWAP_FACE_MODEL)

# Load model for face-restore: now support only ['codeformer']
# if FACE_RESTORE:
download_face_restore_check_ckpts()
upsampler = set_realesrgan()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                 codebook_size=1024,
                                                 n_head=8,
                                                 n_layers=9,
                                                 connect_list=["32", "64", "128", "256"],
                                                 ).to(device)
codeformer_ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
codeformer_checkpoint = torch.load(codeformer_ckpt_path)["params_ema"]
codeformer_net.load_state_dict(codeformer_checkpoint)
codeformer_net.eval()



# get source face list
source_face_list = get_source_face_list(SOURCE_FACE_PATH, face_analyser)
source_face_dict['random'] = source_face_list
# Get source face classification list
# sex: male/female || age: baby(0-3)/kid(3-15)/young(15-35)/middle(35-59)/old(60~)
# male
male_baby_sp = SOURCE_FACE_CLASS + "/male/baby"
source_face_male_baby = get_source_face_list(male_baby_sp, face_analyser)
source_face_dict['male_baby'] = source_face_male_baby
male_kid_sp = SOURCE_FACE_CLASS + "/male/kid"
source_face_male_kid = get_source_face_list(male_kid_sp, face_analyser)
source_face_dict['male_kid'] = source_face_male_kid
male_young_sp = SOURCE_FACE_CLASS + "/male/young"
source_face_male_young = get_source_face_list(male_young_sp, face_analyser)
source_face_dict['male_young'] = source_face_male_young
male_middle_sp = SOURCE_FACE_CLASS + "/male/middle"
source_face_male_middle= get_source_face_list(male_middle_sp, face_analyser)
source_face_dict['male_middle'] = source_face_male_middle
male_old_sp = SOURCE_FACE_CLASS + "/male/old"
source_face_male_old = get_source_face_list(male_old_sp, face_analyser)
source_face_dict['male_old'] = source_face_male_old
# female
female_baby_sp = SOURCE_FACE_CLASS + "/female/baby"
source_face_female_baby = get_source_face_list(female_baby_sp, face_analyser)
source_face_dict['female_baby'] = source_face_female_baby
female_kid_sp = SOURCE_FACE_CLASS + "/female/kid"
source_face_female_kid = get_source_face_list(female_kid_sp, face_analyser)
source_face_dict['female_kid'] = source_face_female_kid
female_young_sp = SOURCE_FACE_CLASS + "/female/young"
source_face_female_young = get_source_face_list(female_young_sp, face_analyser)
source_face_dict['female_young'] = source_face_female_young
female_middle_sp = SOURCE_FACE_CLASS + "/female/middle"
source_face_female_middle = get_source_face_list(female_middle_sp, face_analyser)
source_face_dict['female_middle'] = source_face_female_middle
female_old_sp = SOURCE_FACE_CLASS + "/female/old"
source_face_female_old = get_source_face_list(female_old_sp, face_analyser)
source_face_dict['female_old'] = source_face_female_old


def face_swap_image_process(target_img: np.ndarray, is_restore_face: bool):
    """swap all face in 1 frame"""

    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_BGR2RGB)
    temp_frame = copy.deepcopy(target_img)
    # get face from target image
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_faces = len(source_face_male_baby)
    # check output feature of single face
    print("sample output of face analyser")
    print(f"detection bbox: {target_faces[0].bbox}")
    print(f"sex: {target_faces[0].sex}")
    print(f"age: {target_faces[0].age}")


    if target_faces is not None:

        for i, target_face in enumerate(target_faces):
            # target_id = i
            # source_id = random.randint(0, num_source_faces-1)
            source_id = np_rd.randint(0, num_source_faces-1)
            # temp_frame = swap_face(face_swapper, source_face_list, target_faces, source_id, target_id, temp_frame) # swap all face
            if target_face.sex == "M":
                if target_face.age < 3:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_male_baby[source_id], paste_back=True)
                if 3 <= target_face.age < 15:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_male_kid[source_id], paste_back=True)
                if 15 <= target_face.age < 35:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_male_young[source_id], paste_back=True)
                if 35 <= target_face.age < 59:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_male_middle[source_id], paste_back=True)
                if 59 <= target_face.age:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_male_old[source_id], paste_back=True)
            else:
                if target_face.age < 3:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_female_baby[source_id], paste_back=True)
                if 3 <= target_face.age < 15:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_female_kid[source_id], paste_back=True)
                if 15 <= target_face.age < 35:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_female_young[source_id], paste_back=True)
                if 35 <= target_face.age < 59:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_female_middle[source_id], paste_back=True)
                if 59 <= target_face.age:
                    temp_frame = face_swapper.get(temp_frame, target_face, source_face_female_old[source_id], paste_back=True)

    else:
        print("No target faces found")
    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    if is_restore_face:
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image,
                                        background_enhance=True,
                                        face_upsample=True,
                                        upscale=1,
                                        codeformer_fidelity=0.5,
                                        upsampler=upsampler,
                                        codeformer_net=codeformer_net,
                                        device=device)

        result_image = Image.fromarray(result_image)
        return result_image
    else:
        return result_image


block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown("## AIVISION Privacy Application Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label='upload', type="numpy") #target image
            run_button = gr.Button("Transform")
            with gr.Accordion("Advanced options", open=False):
                face_restore = gr.Checkbox(label="Image Restoration", value=False)
        with gr.Column():
            result_gallery = gr.Image(label='Output')
    ips = [input_image,face_restore]
    run_button.click(fn=face_swap_image_process, inputs=ips,outputs=[result_gallery])

block.launch(share=True)