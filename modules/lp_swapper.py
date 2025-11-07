import torch
import cv2
import numpy as np
import os
from pathlib import Path
import time
import uuid
# util functions
def mask_to_polygon(mask_array, approximation_method=cv2.CHAIN_APPROX_SIMPLE):
  """
  Converts a binary mask to a polygon with the specified approximation method.

  Args:
    mask_array: NumPy array representing the binary mask.
    approximation_method: OpenCV contour approximation method (default: cv2.CHAIN_APPROX_SIMPLE).

  Returns:
    A list of NumPy arrays, where each array contains the (x, y) coordinates of the polygon vertices.
  """

  # Find contours in the mask
  contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, approximation_method)

  # Extract polygon coordinates
  polygons = []
  for contour in contours:
    polygon = contour.squeeze()  # Remove extra dimension
    polygons.append(polygon)

  return polygons

def get_transform_points(contour_points, image, num_points = 10):
    # init params
    lt_min_d = image.shape[0]*image.shape[1]

    lt_point = None
    lt_id = None
    selected_points = list()
    sample_step = int(len(contour_points)/num_points)
    # get the left-top point
    for i, point in enumerate(contour_points):
        # print(point)
        x, y = int(point[0]), int(point[1])
        lt_d = (x ** 2 + y ** 2) ** 0.5

        if lt_d < lt_min_d:
            lt_min_d = lt_d
            lt_point = [x, y]
            lt_id = i
    # get 20 points
    start_point = lt_id - len(contour_points) + 1
    while len(selected_points) < num_points:
        selected_p = contour_points[start_point].tolist()
        selected_points.append(selected_p)
        start_point += sample_step
    return selected_points

def sharpened_img(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Sharpen the image
    img = cv2.filter2D(img, -1, kernel)
    return img
def normalize_img(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img
def lab_img(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #plt.imshow(img)
    #plt.title('lab-method')
    #plt.show()
    return img
def resizeImg(image, height=100):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img
def findTotalContour(img):

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_contour = contours[0]

    for i in range(1, len(contours)):
        total_contour = np.concatenate((total_contour, contours[i]), axis=0)
    hull = cv2.convexHull(total_contour)
    return hull


def getCanny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.GaussianBlur(gray, (3, 3), 2, 2)

    top_border_canny = 800
    bot_border_canny = 200
    bot_border = 10
    top_border = 30
    step = 10
    i = 0
    while i < 10:
        canny = cv2.Canny(binary, bot_border_canny, top_border_canny)
        kernel = np.ones((3, 3), np.uint8)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(top_border_canny, len(contours))
        if bot_border <= len(contours) <= top_border:
            break
        elif len(contours) > top_border:
            top_border_canny += step
        else:
            top_border_canny -= step
        i += 1

    return canny


def get_corners(img):
    img = normalize_img(img)
    img = sharpened_img(img)
    img = lab_img(img)

    try:
        binary_img = getCanny(img)

        max_contour = findTotalContour(binary_img)
        lb_min_d, lt_min_d, rb_min_d, rt_min_d = [img.shape[0] * img.shape[1]] * 4
        for point in max_contour:
            x, y = point[0][0], point[0][1]
            lt_d = (x ** 2 + y ** 2) ** 0.5
            if lt_d < lt_min_d:
                lt_min_d = lt_d
                lt_point = [x, y]

            lb_d = (x ** 2 + (y - img.shape[0]) ** 2) ** 0.5
            if lb_d < lb_min_d:
                lb_min_d = lb_d
                lb_point = [x, y]

            rt_d = ((x - img.shape[1]) ** 2 + y ** 2) ** 0.5
            if rt_d < rt_min_d:
                rt_min_d = rt_d
                rt_point = [x, y]

            rb_d = ((x - img.shape[1]) ** 2 + (y - img.shape[0]) ** 2) ** 0.5
            if rb_d < rb_min_d:
                rb_min_d = rb_d
                rb_point = [x, y]
        points = [lt_point, lb_point, rb_point, rt_point]
        # target_points = np.float32(points)
        return points

    except IndexError:
        return img

def get_corners_bbbox(img):
    lt_point = [0,0]
    lb_point = [0, img.shape[0]]
    rb_point = [img.shape[1], img.shape[0]]
    rt_point = [img.shape[1],0]
    points = [lt_point, lb_point, rb_point, rt_point]

    return points


def get_source_lp_dict(data_root = "data/source_lp"):
    # source_lp_dirs = glob.glob(f"{data_root}/*/")
    # source_lp_path = data_root
    source_lp_dict = dict()
    for type_id in range(7):
        source_lp_path = data_root + "/type" + str(type_id)
        # source_images = []
        for filename in os.listdir(source_lp_path):
            count = 0
            if filename.endswith(".jpg") or filename.endswith(".png"):
                source_img_p = source_lp_path + "/" + filename
                # source_images.append(source_img_p)
                source_img = cv2.imread(source_img_p)
                source_lp_dict[type_id] =  source_img
                count += 1
    return source_lp_dict

def texture_transfer(source_img, target_img, block_size=9, overlap=3):
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

    target_l = target_lab[:, :, 0]

    # Create an empty output image
    output_lab = np.zeros_like(target_lab)
    output_lab[:, :, 0] = target_l  # Use target image's luminance

    for i in range(0, target_img.shape[0], block_size - overlap):
        for j in range(0, target_img.shape[1], block_size - overlap):
            # Extract the current block from the target image
            target_block_l = target_l[i:i + block_size, j:j + block_size]
            # Find the best matching block in the source image based on luminance similarity
            best_match, min_error = None, float('inf')
            for y in range(0, source_img.shape[0] - block_size):
                for x in range(0, source_img.shape[1] - block_size):
                    source_block_l = source_lab[y:y + block_size, x:x + block_size, 0]
                    error = np.sum((target_block_l - source_block_l) ** 2)
                    if error < min_error:
                        min_error = error
                        best_match = source_lab[y:y + block_size, x:x + block_size]
            # Transfer color information from the source block to the output
            output_lab[i:i + block_size, j:j + block_size, 1:] = best_match[:, :, 1:]
    # Convert back to BGR color space
    output_img = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return output_img

def reduce_image_quality(input_image, quality_factor=50):
    # Encode with JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    _, encoded_img = cv2.imencode('.jpg', input_image, encode_param)
    # Decode the compressed image
    reduced_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return reduced_image

def blend_images(background, foreground, mask):
    #load images

    # resize mask if necessary
    if mask.shape[:2] != foreground.shape[:2]:
        mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

    # Norm mask to 0.0 - 1.0
    mask = mask.astype(np.float32) / 255.0

    # Create an inverted mask
    inv_mask = 1- mask

    # Blend foreground and background
    blended_image = (foreground * mask[:, :, np.newaxis]) + (background * inv_mask[:, :, np.newaxis])

    # save the results
    return blended_image

def get_transform_lp(target_img, source_img, image_name="", image_log=False):
    """return the image for matching
    input: target_img: np_array, source_img: np_array
    return: image for blending"""

    # Get transformation points
    corners = get_corners_bbbox(target_img)
    # Adding noise
    # hs,ws = source_img.shape[1], source_img.shape[0]
    # resized_target = cv2.resize(target_img,(hs,ws),interpolation = cv2.INTER_LINEAR)
    # gray_src = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    # gray_dest = cv2.cvtColor(resized_target, cv2.COLOR_BGR2GRAY)
    # noise_stddev = np.std(gray_dest - gray_src)
    # noisy_src = source_img + np.random.normal(0, noise_stddev, source_img.shape).astype(np.uint8)

    # texture transfer
    # new_source_img = texture_transfer(resized_target, source_img)

    if len(corners) == 4:
        target_points = np.float32(corners)
        # Create a mask for blending
        mask = np.zeros_like(target_img[:, :, 0])
        cv2.fillConvexPoly(mask, np.array(corners, dtype=np.int32), 255)

        # Get center of image
        center_x = int((target_img.shape[1]) / 2)
        center_y = int((target_img.shape[0]) / 2)
        center = (center_x, center_y)

        # get transform image
        source_points = np.float32(
            [[0, 0], [0, source_img.shape[0]], [source_img.shape[1], source_img.shape[0]], [source_img.shape[1], 0]])
        M = cv2.getPerspectiveTransform(source_points, target_points)

        transform_img = cv2.warpPerspective(source_img, M, (target_img.shape[1], target_img.shape[0]))

        reduce_transform_img = reduce_image_quality(transform_img, quality_factor=50)

        # Blending image
        # print("transform_img:",transform_img.shape)
        # print("target_points:",target_img.shape)
        # print("mask:",mask.shape)
        # print("center:",center)
        blend_crop = cv2.seamlessClone(reduce_transform_img, target_img, mask, center, cv2.NORMAL_CLONE)
        # blend_crop = blend_images(target_img, transform_img, mask)
        if image_log:
            for c in corners:
                cv2.circle(target_img, tuple(c), 5, (0, 0, 255), 2)
            # save log images
            count = image_name
            save_p = "output/yolo_segment/"
            transform_save_p = save_p + str(count) + "_transform.jpg"
            mask_save_p = save_p + str(count) + "_mask.jpg"
            blend_save_p = save_p + str(count) + "_blend.jpg"
            crop_save_p = save_p + str(count) + "_points.jpg"

            cv2.imwrite(crop_save_p, target_img)
            cv2.imwrite(transform_save_p, transform_img)
            cv2.imwrite(mask_save_p, mask)
            cv2.imwrite(blend_save_p, blend_crop)
    else:
        blend_crop = target_img

    return blend_crop


# padding to remove number
def lp_padding(target_image):
    """hidding number plate for a better conner detection"""
    # fill a mask on the center of crop image
    padding_offset = int(target_image.shape[0] / 5)

    p_y0 = padding_offset
    p_x0 = padding_offset
    p_y1 = target_image.shape[0] - padding_offset
    p_x1 = target_image.shape[1] - padding_offset
    padding_value = int(np.mean(target_image[p_x0, p_y0]))

    padding_region = target_image[p_y0:p_y1, p_x0:p_x1]
    # print(padding_value)
    padding_mask = np.ones(padding_region.shape[:2], dtype=np.uint8) * padding_value
    target_filled = overlay_image(target_image, padding_mask, pos=(p_x0, p_y0))
    return target_filled

def overlay_image(original_frame,foreground,pos=(0,0),trans=1.0):
    background = original_frame.copy()
    #get position and crop pasting area if needed
    x = pos[0]
    y = pos[1]
    bgWidth = background.shape[1]
    bgHeight = background.shape[0]
    frWidth = foreground.shape[1]
    frHeight = foreground.shape[0]
    # check the size
    if x + frWidth > bgWidth or y + frHeight > bgHeight:
        raise ValueError("Target area exceeds the bounds of the background image.")
    # convert the image to RGBA
    foreground_img = cv2.cvtColor(foreground,cv2.COLOR_BGR2BGRA)
    # print(foreground_img.shape)
    # Create a transparency mask
    alpha_mask = np.ones(foreground_img.shape[:2], dtype=np.uint8)*255*trans

    # Extract the relevant region from the foreground image
    foreground_region = foreground_img[:foreground_img.shape[0], :foreground_img.shape[1], :]

    # Blend the images using alpha blending within the target area
    for c in range(0,3):
        background[y:y+foreground_region.shape[0], x:x+foreground_region.shape[1],c] = (1-alpha_mask/255.0)*background[y:y+foreground_region.shape[0],x:x+foreground_region.shape[1],c] + alpha_mask/255.0*foreground_region[:,:,c]
    # Combine the image
    # result = cv2.addWeighted(background, 1, foreground_img,1,0)
    return background

def swap_lp_image(detection_engine, input_image, source_lp_dict):
    start_detect_time = time.time()
    results = detection_engine.predict(input_image, save=False, stream=True, show=False, imgsz=640, verbose=False)
    stop_detect_time = time.time()
    # print("inference_time: ", (stop_detect_time-start_detect_time))
    crop_offset = 0
    frame = input_image.copy()
    for result in results:
        clss = result.boxes.cpu().cls
        boxes = result.boxes.cpu().xyxy
        # frame = result.orig_img.copy()
        # print("number of detected lp: {}".format(len(boxes)))
        for cls, box in zip(clss, boxes):
            height, width = result.orig_img.shape[:2]
            crop_x0 = int(box[0]) - crop_offset if (int(box[0]) - crop_offset > 0) else 0
            crop_y0 = int(box[1]) - crop_offset if (int(box[1]) - crop_offset > 0) else 0
            crop_x1 = int(box[2]) + crop_offset if (int(box[2]) + crop_offset < width) else width
            crop_y1 = int(box[3]) + crop_offset if (int(box[3]) + crop_offset < height) else height
            #print([crop_y0, crop_y1, crop_x0, crop_x1])
            crop_plate = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            # target_filled = lp_padding(crop_plate)
            #print(target_filled.shape)
            type_id = int(cls)
            if type_id <= 6:
                transform_image = get_transform_lp(crop_plate, source_lp_dict[type_id], image_name="")
            else:
                transform_image = get_transform_lp(crop_plate, source_lp_dict[0], image_name="")
            #print(transform_image.shape)
            frame = overlay_image(frame, transform_image, pos=(crop_x0, crop_y0))
    return frame


def swap_lp_video_process_all(target_video, detection_engine, source_lp_dict):
    """input:
            target_video: video for lp swapping
            detection_engine: detection model loaded
            source_lp_dict: dictionary for source lp
        output:
            input_video_name: processed input video for streaming
            output_video_name: processed output video for streaming
            first_frame_name: first frame video for streaming"""

    cap = cv2.VideoCapture(target_video)

    # Setup the writer
    input_video_name = Path(target_video).name.split(".")[0]
    input_video_name = input_video_name.replace(" ", "")
    writer_video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_writer_name = f"output/stream_output/input_{input_video_name}_{uuid.uuid4()}.mp4"
    first_frame_writer_name = f"output/stream_output/first_frame_{input_video_name}_{uuid.uuid4()}.mp4"
    output_writer_name = f"output/stream_output/first_frame_{input_video_name}_{uuid.uuid4()}.mp4"
    input_writer = None
    first_frame_writer = None
    output_writer = None
    count = 0
    while True:
        ret, cap_frame = cap.read()

        if ret:
            original_height, original_width, _ = cap_frame.shape
            # Specify the desired height
            desired_height = 600

            # Calculate the aspect ratio
            aspect_ratio = original_width / original_height

            # Calculate the desired width based on the aspect ratio and desired height
            desired_width = int(desired_height * aspect_ratio)

            # Resize the image
            resized_frame = cv2.resize(cap_frame, (desired_width, desired_height))

            if input_writer is None:
                input_writer = cv2.VideoWriter(input_writer_name, writer_video_codec, input_fps,
                                               (desired_width, desired_height))
                first_frame_writer = cv2.VideoWriter(first_frame_writer_name, writer_video_codec, input_fps,
                                                     (desired_width, desired_height))
                output_writer = cv2.VideoWriter(output_writer_name, writer_video_codec, input_fps,
                                                    (desired_width, desired_height))

            swapped_frame = swap_lp_image(detection_engine, resized_frame, source_lp_dict)

            # save frames
            input_writer.write(resized_frame)
            output_writer.write(swapped_frame)
            if count == 0:
                first_frame_writer.write(resized_frame)

            count += 1
        else:
            break
    cap.release()
    input_writer.release()
    first_frame_writer.release()
    output_writer.release()

    return input_writer_name, first_frame_writer_name, output_writer_name

# functions for lp swap streaming
def preprocess_video_for_lp(target_video):
    cap = cv2.VideoCapture(target_video)

    # Setup the writer
    input_video_name = Path(target_video).name.split(".")[0]
    input_video_name = input_video_name.replace(" ", "")

    writer_video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))

    input_writer_name = f"output/stream_output/input_{input_video_name}_{uuid.uuid4()}.mp4"
    first_frame_writer_name = f"output/stream_output/first_frame_{input_video_name}_{uuid.uuid4()}.mp4"

    input_writer = None
    first_frame_writer = None

    count = 0
    while True:
        ret, cap_frame = cap.read()

        if ret:
            original_height, original_width, _ = cap_frame.shape
            # Specify the desired height
            desired_height = 600

            # Calculate the aspect ratio
            aspect_ratio = original_width / original_height

            # Calculate the desired width based on the aspect ratio and desired height
            desired_width = int(desired_height * aspect_ratio)

            # Resize the image
            resized_frame = cv2.resize(cap_frame, (desired_width, desired_height))
            if input_writer is None:
                input_writer = cv2.VideoWriter(input_writer_name, writer_video_codec, input_fps,
                                               (desired_width, desired_height))
                first_frame_writer = cv2.VideoWriter(first_frame_writer_name, writer_video_codec, input_fps,
                                                     (desired_width, desired_height))

            # save frames
            input_writer.write(resized_frame)

            if count == 0:
                first_frame_writer.write(resized_frame)

            count += 1
        else:
            break

    cap.release()
    input_writer.release()
    first_frame_writer.release()

    return input_writer_name, first_frame_writer_name