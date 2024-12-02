
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

# Video processing functions

def get_temp_directory_path(target_path):
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    target_frame_directory = os.path.join(target_directory_path, target_name)
    if not os.path.exists(target_frame_directory):
        os.mkdir(target_frame_directory)
    return target_frame_directory

def get_temp_frame_paths(target_path):
    temp_directory_path = get_temp_directory_path(target_path)

    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.jpg')))

def extract_frame(target_path):
    save_frame_path = get_temp_directory_path(target_path)
    # extract with cv2 here
    cap = cv2.VideoCapture(target_path)
    # frame counter
    count = 0
    # check if frame was extracted

    while True:
        ret, frame = cap.read()
        if ret:
            # frame_path = save_frame_path + "/" + "image_000000" + str(count) + ".jpg"
            if count < 10:
                frame_path = save_frame_path + "/" + "image_000000" + str(count) + ".jpg"
            if 10 <= count < 100:
                frame_path = save_frame_path + "/" + "image_00000" + str(count) + ".jpg"
            if 100 <= count < 1000:
                frame_path = save_frame_path + "/" + "image_0000" + str(count) + ".jpg"
            if 1000<= count < 10000:
                frame_path = save_frame_path + "/" + "image_000" + str(count) + ".jpg"
            if 10000 <= count < 100000:
                frame_path = save_frame_path + "/" + "image_00" + str(count) + ".jpg"
            if 100000 <= count < 1000000:
                frame_path = save_frame_path + "/" + "image_0" + str(count) + ".jpg"
            if 1000000 <= count:
                frame_path = save_frame_path + "/" + "image_" + str(count) + ".jpg"
            # save the frame with frame-count
            cv2.imwrite(str(frame_path), frame)
            count += 1
        else:
            break

def find_cluster_centroids(embeddings, max_k=10):
    inertia = []
    cluster_centroids = []
    k_clusters = range(1, max_k+1)

    for k in k_clusters:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    optimal_centroids = cluster_centroids[diffs.index(max(diffs)) + 1]['centroids']

    return optimal_centroids

def find_closest_centroid(centroids: list, normed_face_embedding):
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)

        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None

def get_unique_faces_from_target_video(target_video_path,face_analyser, source_face_dict) -> Any:
    try:
        source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []

        # extract faces from the given video and get the list of image
        extract_frame(target_video_path)
        temp_frame_paths = get_temp_frame_paths(target_video_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(face_analyser, temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append({'frame': i, 'faces':many_faces, 'location': temp_frame_path})
            i += 1
        centroids = find_cluster_centroids(face_embeddings)
        # Scan frames again to determine the id of each face
        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids,face.normed_embedding)
                face['target_centroid'] = closest_centroid_index
        # determine the map face to swap
        # the len of centroids is the number of faces in the video
        for i in range(len(centroids)):
            source_target_map.append({
                'id':i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings,desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})
            source_target_map[i]['target_faces_in_frame'] = temp

        # default_target_face(source_target_map)
        for map_face in source_target_map:
            best_face = None
            best_frame = None
            for frame in map_face["target_faces_in_frame"]:
                if len(frame['faces']) > 0:
                    best_face = frame['faces'][0]
                    best_frame = frame
                    break

            for frame in map_face['target_faces_in_frame']:
                for face in frame['faces']:
                    if face['det_score'] > best_face['det_score']:
                        best_face = face
                        best_frame = frame
            x_min, y_min, x_max, y_max = best_face['bbox']

            target_frame = cv2.imread(best_frame['location'])
            map_face['target'] = {
                'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                'face': best_face
            }
            # find proper source face for the target face
            max_source_face = len(source_face_dict['male_baby'])
            source_id = np_rd.randint(0, max_source_face-1)
            best_face_gender = int(map_face['target']['face']['gender'])
            best_face_age = int(map_face['target']['face']['age'])
            if best_face_gender==1: # male
                if best_face_age < 3:
                    map_face['source'] = source_face_dict['male_baby'][source_id]
                if 3 <= best_face_age < 15:
                    map_face['source'] = source_face_dict['male_kid'][source_id]
                if 15 <= best_face_age < 35:
                    map_face['source'] = source_face_dict['male_young'][source_id]
                if 35 <= best_face_age < 59:
                    map_face['source'] = source_face_dict['male_middle'][source_id]
                if 59 <= best_face_age:
                    map_face['source'] = source_face_dict['male_old'][source_id]
            else:
                if best_face_age < 3:
                    map_face['source'] = source_face_dict['female_baby'][source_id]
                if 3 <= best_face_age < 15:
                    map_face['source'] = source_face_dict['female_kid'][source_id]
                if 15 <= best_face_age < 35:
                    map_face['source'] = source_face_dict['female_young'][source_id]
                if 35 <= best_face_age < 59:
                    map_face['source'] = source_face_dict['female_middle'][source_id]
                if 59 <= best_face_age:
                    map_face['source'] = source_face_dict['female_old'][source_id]

        return source_target_map, frame_face_embeddings
    except ValueError:
        return None

def default_target_face(source_target_map):
    for map_face in source_target_map:
        best_face = None
        best_frame = None
        for frame in map_face["target_faces_in_frame"]:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map_face['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame
        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        map_face['target'] = {
            'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
            'face': best_face
        }










