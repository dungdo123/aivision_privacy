from Demos.RegCreateKeyTransacted import transacted_subkeys

from modules.face_swapper import *
from modules.face_restoration import *
import onnxruntime as ort
import torch
from multiprocessing import Process, Queue
import random


# Input setting
# input params setting
SOURCE_FACE_PATH = "data/source_faces"
SOURCE_FACE_CLASS = "data/source_face_classification"
SWAP_FACE_MODEL = "models/swapface/aivision_swapface_v1.onnx"
FACE_RESTORE = False
source_face_dict = dict()

# load machine default available providers
# providers = ort.get_available_providers()
providers = ['CUDAExecutionProvider']
# print(f"list of providers{providers}")
# load the face analyser
face_analyser = get_face_analyser(providers)

# load face_swapper
# model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), SWAP_FACE_MODEL)
# face_swapper = get_face_swap_model(SWAP_FACE_MODEL)

# get source face list
source_face_list = get_source_face_list(SOURCE_FACE_PATH, face_analyser)

# face swap worker
def faceswap_worker(model_path, input_queue, output_queue, static_frame):
    # load model
    print("input queue:{}".format(input_queue.get()))
    print("model path:{}".format(model_path))
    face_swapper = get_face_swap_model(model_path)

    source_id = random.randint(0, 19)
    while True:
        if input_queue.empty():
            print("input is empty")

        else:
            target_face = input_queue.get()
            # if target_face is None:
            #     print("input is None")
            #     break
            # print(target_face)
            # temp_frame = face_swapper.get(static_frame, target_face, source_face_list[source_id], paste_back=True)
            #
            output_queue.put(target_face)

if __name__ == "__main__":
    # number of worker
    num_workers = 2

    # create input and output queue
    input_q = Queue()
    output_q = Queue()

    # get the list of target face
    input_image_path = "test_data/Korean_image/1.png"
    test_image = cv2.imread(input_image_path)

    original_height, original_width, _ = test_image.shape
    # Specify the desired height
    desired_height = 400

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate the desired width based on the aspect ratio and desired height
    desired_width = int(desired_height * aspect_ratio)
    resized_frame = cv2.resize(test_image, (desired_width, desired_height))

    # Create worker process
    processes = []
    for _ in range(num_workers):
        process = Process(target=faceswap_worker, args=(SWAP_FACE_MODEL, input_q, output_q,resized_frame))
        processes.append(process)
        process.start()
    # target_faces = get_many_faces(face_analyser, resized_frame)


    # Signal to stop worker
    # for _ in range(num_workers):
    #     input_q.put(None)
    # Wait for all workers to finish

    # Get inference results from output queue
    results = []
    count = 0
    while count < 100:
        target_faces = [1, 2, 3, 4]
        for t in target_faces:
            # print("target face put in queue: {}".format(t))

            input_q.put(t)


        for process in processes:
            process.join()

        if output_q.empty():
            break
        results.append(output_q.get(timeout=1))
        print(results)
        count +=1
    # for i,result in enumerate(results):
    #     save_output_name = "output/output" + str(i) + ".jpg"
    #     cv2.imwrite(save_output_name, result)



