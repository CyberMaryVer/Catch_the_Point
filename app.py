import cv2.cv2 as cv2
from time import time
import imutils
import numpy as np
from datetime import datetime
from mp_predictor import MpipePredictor, get_updated_keypoints
from visualization import visualize_keypoints
from geometry import get_distance

KEYPOINTS_FOR_GAME = ["nose", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow",
                      "right_elbow", "left_knee", "right_knee"]
WITH_SKELETON = False
GAMES_NUMBER = 4
WINDOW_NAME = "Catch the point!"
SCALE = 1.4


def draw_nose(img, keypoints):

    if keypoints is None:
        return img
    if len(keypoints) == 17:
        keypoints = get_updated_keypoints(keypoints)

    center = (keypoints["nose"][0], keypoints["nose"][1])
    hcenter = (keypoints["head_center"][0], keypoints["head_center"][1])
    radius = int(get_distance(center, hcenter) / 2)
    color = (0, 0, 255)
    img = cv2.circle(img=img,
                     center=center,
                     radius=radius,
                     color=color,
                     thickness=-1,
                     lineType=cv2.LINE_AA)
    return img


def draw_box_with_text(img, text=None, edge_color=(255, 255, 255), border=2, mode=0):
    """
    draws box around
    """
    # width, height = img.shape[1::-1]
    # scale = max(width, height) / 400
    font_scale, font_thickness = .8, 2
    font_color = (0, 0, 0)

    if mode == 0:  # standard mode
        img = cv2.copyMakeBorder(img, 10 * border, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)

    elif mode == 1:  # low vision
        img = cv2.copyMakeBorder(img, 10 * border, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)
        font_scale, font_thickness = 1.6, 2

    if text is not None:
        x = y = border
        img = cv2.putText(img, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
                          lineType=cv2.LINE_AA)

    return img


def mp_pose_game(img, keypoints, shape, game_state=0, active_point=None, catch_point=None, radius=1, color=None):
    w, h = shape
    color_end = (0, 255, 0)
    color_start = color
    txt = ""

    if game_state == 0:
        active_point = np.random.choice(KEYPOINTS_FOR_GAME)
        xx, yy = np.random.randint(0, w), np.random.randint(0, h)
        catch_point = (xx, yy)
        txt = f"GAME STARTED\nUSE YOUR {active_point.upper().replace('_', ' ')}!"
        print(txt)

    game_state = 1 if game_state == 0 else game_state
    xx, yy = catch_point
    xa, ya, _ = keypoints[active_point]
    # radius = int(16) + np.random.choice([-1, 0, 1])
    success_radius = radius
    img = cv2.circle(img, (xa, ya), 8, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    if abs(xx - xa) < success_radius and abs(yy - ya) < success_radius:
        game_state = 2
        txt = f"CONGRATULATIONS!!!!"
        color = color_end
    else:
        color = color_start
        txt = f"CATCH THE BLUE POINT OR IT WILL CATCH YOU!!"

    img = cv2.circle(img, (xx, yy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    img = draw_box_with_text(img, txt, edge_color=(255, 255, 255), border=6)

    return img, active_point, game_state, catch_point, txt


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    shape = (cam.get(3), cam.get(4))
    scale = max(shape) / 400 * SCALE
    scaled_width = int(shape[0] * SCALE)

    kps = None
    game = 0
    score = 0
    point = None
    cpoint = None
    radius = 1
    success = 0
    c1, c2, c3 = 255, 0, 0
    txt = ""
    frame = frame_ = None
    predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)

    while cv2.waitKey(1) != 27 or success == GAMES_NUMBER:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_ = frame.copy()  # for final photo
        frame = imutils.resize(frame, width=scaled_width)

        if not success == GAMES_NUMBER:
            try:
                outputs = predictor.get_keypoints(frame)
                color = (int(c1), int(c2), int(c3))
                kps = get_updated_keypoints(outputs)
                radius = int(radius)
                frame = visualize_keypoints(kps, frame, skeleton=2, dict_is_updated=True, threshold=.7, scale=scale)
                frame, point, game, cpoint, txt = mp_pose_game(img=frame, keypoints=kps, shape=shape, game_state=game,
                                                               active_point=point, catch_point=cpoint, radius=radius,
                                                               color=color)
                radius += np.random.choice([0, 1, 0])
                c1 -= .4
                c3 += .4

                if int(c1) == 0:
                    txt = f"GAME OVER! YOUR SCORE {score}"
                    cam.release()
                    break

                if game == 2:
                    print("YOU CATCH IT!")
                    success += 1
                    score += 500 // radius
                    radius = min(6, radius - 10)
                    game = 0

                if success == GAMES_NUMBER:
                    txt = f"YOU WIN!!! YOUR SCORE {score}"
                    cam.release()
                    break

            except Exception as e:
                txt = "Try to stay visible for the camera"
                frame = draw_box_with_text(frame, txt, edge_color=(255, 255, 255), border=6)

            cv2.imshow(WINDOW_NAME, frame)

    cam.release()
    cv2.waitKey(5)

    if "WIN" in txt:
        frame_color = (0, 255, 0)
    elif "OVER" in txt:
        frame_color = (188, 188, 188)
    else:
        txt = f"YOU EXIT THE GAME. YOU SCORE {score}"
        frame_color = (100, 100, 200)

    print(txt)
    color = (int(c1), int(c2), int(c3))

    final_photo = frame if WITH_SKELETON else frame_  # use clean frame if WITH_SKELETON == False
    final_overlay = final_photo.copy()
    final_overlay = cv2.circle(final_overlay, cpoint, radius, color, -1, cv2.LINE_AA)
    final_photo = cv2.addWeighted(final_photo, .7, final_overlay, .3, 1)
    final_photo = draw_box_with_text(final_photo, txt, edge_color=frame_color, border=6)

    final_photo = improve_photo(final_photo)

    cv2.imshow(WINDOW_NAME, final_photo)
    date_and_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
    img_name = f"result_{date_and_time}.jpg"
    img_path = "gallery/" + img_name
    cv2.imwrite(img_path, final_photo)
    cv2.waitKey(0)

    while cv2.waitKey(1) != 27:
        pass
    cv2.destroyAllWindows()

def improve_photo(img):
    img_ = img.copy()
    img_ = cv2.detailEnhance(img_, sigma_s=20, sigma_r=0.15)
    img_ = cv2.edgePreservingFilter(img_, flags=1, sigma_s=60, sigma_r=0.15)
    img_ = cv2.stylization(img_, sigma_s=95, sigma_r=0.95)
    img = cv2.addWeighted(img, .8, img_, .7, .5)

    return img


if __name__ == "__main__":
    main()
    # im = cv2.imread("tests/out.jpg")
    # im = improve_photo(im)
    # cv2.imshow("", im)
    # cv2.waitKey(0)
