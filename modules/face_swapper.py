
import os.path
from email.policy import default
from multiprocessing.managers import Value
from random import random

import cv2
import insightface
import numpy as np
from typing import Any
from sklearn.cluster import KMeans

from modules.utils import *
import glob
from tqdm import tqdm
from numpy import random as np_rd
from PIL import Image
# utils functions
def get_source_face_list(source_face_path, face_analyser):
    source_image_list = []
    source_face_list = []
    for face_img in os.listdir(source_face_path):
        if face_img.endswith(".jpg") or face_img.endswith(".png"):
            face_path = source_face_path + "/" + face_img
            source_image = Image.open(face_path)
            source_image_list.append(source_image)
    for img in source_image_list:
        s_face = get_many_faces(face_analyser, cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        # print(type(s_face[0]))
        source_face_list.append(s_face[0])

    return source_face_list


def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(providers, det_size=(320,320)):
    face_analyzer = insightface.app.FaceAnalysis(name="face_analysis_models", root="./", providers=providers)
    face_analyzer.prepare(ctx_id=0, det_size=det_size)
    return face_analyzer


def get_one_face(face_analyzer, frame: np.ndarray):
    """get the face at the left"""
    face = face_analyzer.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame:np.ndarray):
    """get faces from left to right by order"""
    try:
        faces = face_analyser.get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """paste source_face on target image"""
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

# adding face analyser functions
def get_unique_faces_from_target_image(target_image_path) -> Any:
    """attach the face id into the image"""
    try:
        source_target_map = []
        target_frame = cv2.imread(target_image_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            source_target_map.append({
                'id': i,
                'target' : {
                    'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                    'face': face
                }
            })
            i = i + 1
    except ValueError:
        return None
