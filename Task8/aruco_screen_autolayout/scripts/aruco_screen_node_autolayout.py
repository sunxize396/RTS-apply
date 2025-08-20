#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    import rosbag
except Exception:
    rosbag = None

def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def draw_grid_image(ids_grid, dictionary_name="DICT_4X4_50", cell_px=300, pad_px=40, bg=255):
    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name)
    adict = aruco.getPredefinedDictionary(dict_id)
    rows = len(ids_grid)
    cols = len(ids_grid[0]) if rows > 0 else 0
    H = rows * cell_px + (rows + 1) * pad_px
    W = cols * cell_px + (cols + 1) * pad_px
    canvas = np.full((H, W), bg, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            mid = ids_grid[r][c]
            y0 = pad_px + r * (cell_px + pad_px)
            x0 = pad_px + c * (cell_px + pad_px)
            if mid is None or mid < 0:
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_px, y0 + cell_px), 0, 2)
                continue
            marker = np.zeros((cell_px, cell_px), dtype=np.uint8)
            cv2.aruco.drawMarker(adict, int(mid), cell_px, marker, 1)
            canvas[y0:y0 + cell_px, x0:x0 + cell_px] = marker
    return canvas

def project_points(H, pts):
    pts = np.asarray(pts, np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts),1), np.float32)])
    w = (H @ pts_h.T).T
    w = w[:, :2] / w[:, 2:3]
    return w

def layout_from_observation(obs_centers, H_or_none, rows=2, cols=4):
    if not obs_centers:
        return [[-1 for _ in range(cols)] for __ in range(rows)]
    ids = list(obs_centers.keys())
    pts = np.array([obs_centers[i] for i in ids], np.float32)
    if H_or_none is not None:
        pts = project_points(H_or_none, pts)
    idx = np.argsort(pts[:,1])
    ids_sorted = [ids[i] for i in idx]
    pts_sorted = pts[idx]
    parts = []
    N = len(ids_sorted)
    for r in range(rows):
        a = int(r * N / rows); b = int((r+1) * N / rows)
        parts.append((ids_sorted[a:b], pts_sorted[a:b]))
    grid = [[-1 for _ in range(cols)] for __ in range(rows)]
    for r in range(rows):
        ids_r, pts_r = parts[r]
        if len(ids_r) == 0: continue
        order_x = np.argsort(pts_r[:,0])
        ids_row_sorted = [ids_r[i] for i in order_x]
        for c in range(min(cols, len(ids_row_sorted))):
            grid[r][c] = int(ids_row_sorted[c])
    return grid

class ArucoScreenNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_rect_color")
        self.rosbag_path = rospy.get_param("~rosbag_path", "")
        self.dictionary_name = rospy.get_param("~dictionary", "DICT_4X4_50")
        self.expected_grid = rospy.get_param("~expected_grid", [[0,1],[2,3],[4,5],[6,7]])
        self.auto_layout = rospy.get_param("~auto_layout", True)
        self.auto_rows   = int(rospy.get_param("~auto_rows", 2))
        self.auto_cols   = int(rospy.get_param("~auto_cols", 4))

        self.pub_rect = rospy.Publisher("~rectified", Image, queue_size=2, latch=True)
        self.pub_recon = rospy.Publisher("~reconstructed", Image, queue_size=2, latch=True)
        self.pub_rawimage = rospy.Publisher("~rawimage", Image, queue_size=2, latch=True)

        aruco = cv2.aruco
        dict_id = getattr(aruco, self.dictionary_name)
        self.adict = aruco.getPredefinedDictionary(dict_id)
        self.params = aruco.DetectorParameters_create()

        self.cell_px = rospy.get_param("~cell_px", 300)
        self.pad_px = rospy.get_param("~pad_px", 40)
        self.bg = rospy.get_param("~bg_level", 255)

        H = self.auto_rows * self.cell_px + (self.auto_rows + 1) * self.pad_px
        W = self.auto_cols * self.cell_px + (self.auto_cols + 1) * self.pad_px
        self.canon_size = (int(W), int(H))

        self.last_good_H = None

        if self.rosbag_path:
            if rosbag is None: raise RuntimeError("rosbag API not available")
            self.run_rosbag_loop()
        else:
            self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

    def run_rosbag_loop(self):
        rate = rospy.Rate(rospy.get_param("~bag_rate_hz", 30.0))
        with rosbag.Bag(self.rosbag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                if rospy.is_shutdown(): break
                self.image_cb(msg); rate.sleep()

    def image_cb(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = ensure_gray(cv_img)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.adict, parameters=self.params)

        debug_viz = cv_img.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_viz, corners, ids)

        obs_centers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten().tolist()):
                pts = corners[i].reshape(4, 2)
                cxy = pts.mean(axis=0)
                obs_centers[int(marker_id)] = (float(cxy[0]), float(cxy[1]))

        # Estimate H from temporary ordering for stabilization
        if len(obs_centers) >= 4:
            temp_grid = layout_from_observation(obs_centers, None, self.auto_rows, self.auto_cols)
            canon_centers = {}
            for r in range(self.auto_rows):
                for c in range(self.auto_cols):
                    cx = self.pad_px + c*(self.cell_px+self.pad_px) + self.cell_px/2.0
                    cy = self.pad_px + r*(self.cell_px+self.pad_px) + self.cell_px/2.0
                    canon_centers[(r,c)] = (cx,cy)
            src_pts, dst_pts = [], []
            for r in range(self.auto_rows):
                for c in range(self.auto_cols):
                    mid = temp_grid[r][c]
                    if mid is not None and mid >= 0 and mid in obs_centers:
                        src_pts.append(obs_centers[mid]); dst_pts.append(canon_centers[(r,c)])
            if len(src_pts) >= 4:
                H_now, _ = cv2.findHomography(np.array(src_pts,np.float32),
                                              np.array(dst_pts,np.float32),
                                              method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if H_now is not None:
                    self.last_good_H = H_now

        if self.last_good_H is not None:
            rectified = cv2.warpPerspective(cv_img, self.last_good_H, self.canon_size)
            self.pub_rect.publish(self.bridge.cv2_to_imgmsg(rectified, encoding="bgr8"))

        if self.auto_layout and ids is not None and len(ids) > 0:
            grid = layout_from_observation(obs_centers, self.last_good_H, self.auto_rows, self.auto_cols)
        else:
            rows = len(self.expected_grid); cols = len(self.expected_grid[0]) if rows>0 else 0
            grid = [[-1 for _ in range(cols)] for __ in range(rows)]
            if ids is not None:
                id_to_rc = {}
                for r,row in enumerate(self.expected_grid):
                    for c,mid in enumerate(row):
                        if int(mid) >= 0:
                            id_to_rc[int(mid)] = (r,c)
                for mid in ids.flatten().tolist():
                    if mid in id_to_rc:
                        r,c = id_to_rc[mid]; grid[r][c] = mid

        recon = draw_grid_image(grid, self.dictionary_name, self.cell_px, self.pad_px, self.bg)
        self.pub_recon.publish(self.bridge.cv2_to_imgmsg(recon, encoding="mono8"))
        self.pub_rawimage.publish(self.bridge.cv2_to_imgmsg(debug_viz, encoding="bgr8"))

def main():
    rospy.init_node("aruco_screen_node")
    ArucoScreenNode()
    rospy.spin()

if __name__ == "__main__":
    main()

