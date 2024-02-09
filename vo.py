import open3d as o3d
import numpy as np
import cv2 as cv
import sys
import os
import argparse
import glob
import multiprocessing as mp

'''
'''


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.frame_paths = sorted(
            list(glob.glob(os.path.join(args.input, '*.png'))))
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)
        self.P = np.hstack((self.R, -self.t))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()

        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    # TODO:insert new camera pose here using vis.add_geometry()
                    # set points
                    # image coordinate
                    points = np.array(
                        [[0, 0, 1], [640, 0, 1], [640, 360, 1], [0, 360, 1]])
                    # transform to world coordinate
                    points = np.matmul(np.linalg.pinv(self.K), points.T)
                    points = t + np.matmul(R, points)
                    points = points.T
                    # insert camera center
                    points = np.insert(points, 0, t.T, 0)
                    # set lines
                    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                             [1, 2], [1, 4], [2, 3], [3, 4]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    # set color
                    colors = [[0, 1, 0] for i in range(len(lines))]
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    # draw pose
                    vis.add_geometry(line_set)
            except:
                pass

            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        first_time = True
        for frame_path in self.frame_paths[1:]:
            img = cv.imread(frame_path)
            img_query = cv.imread(
                self.frame_paths[self.frame_paths.index(frame_path)-1])
            # TODO: compute camera pose here
            # Initiate ORB detector
            orb = cv.ORB_create()
            if(first_time == True):
                # find keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img_query, None)
                kp2, des2 = orb.detectAndCompute(img, None)
                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                # Get the image points from the matches
                points1 = np.empty((0, 2))
                points2 = np.empty((0, 2))
                index_last_pair = []  # keep index of pairs in k-1,k
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    index_last_pair.append(img_index)
                    points1 = np.vstack(
                        (points1, kp1[query_index].pt))
                    points2 = np.vstack(
                        (points2, kp2[img_index].pt))
                # keep descriptor of pairs in k-1,k
                self.des_last_pair = des2[index_last_pair]
                # undistortion
                points1 = cv.undistortPoints(
                    points1, self.K, self.dist, None, self.K)
                points2 = cv.undistortPoints(
                    points2, self.K, self.dist, None, self.K)
                # find Essential matrix
                E, inlier = cv.findEssentialMat(points1, points2, self.K)
                # decompose E to get pose
                _, R, t, inlier = cv.recoverPose(E, points1, points2, self.K)
                # process inlier
                index_inlier = np.argwhere(inlier.flatten()).flatten()
                points1 = points1[index_inlier]
                points2 = points2[index_inlier]
                self.des_last_pair = self.des_last_pair[index_inlier]
                # form projection matrix
                t = -t
                P = np.hstack((R, t))
                # calculate X
                self.last_X = cv.triangulatePoints(self.P, P, points1, points2)
                self.last_X = (self.last_X/self.last_X[-1]).T
                # record the pose of the first pair
                self.P = P
                self.t = t
                self.R = R

                first_time = False

            else:
                # find keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img_query, None)
                kp2, des2 = orb.detectAndCompute(img, None)
                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                # Get the image points from the matches
                # Get the matched descriptor of current pair
                points1 = np.empty((0, 2))
                points2 = np.empty((0, 2))
                index_current_pair = []
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    index_current_pair.append(img_index)
                    points1 = np.vstack((points1, kp1[query_index].pt))
                    points2 = np.vstack((points2, kp2[img_index].pt))
                # keep descriptor of pairs in k,k+1
                des2 = des2[index_current_pair]
                # undistortion
                points1 = cv.undistortPoints(
                    points1, self.K, self.dist, None, self.K)
                points2 = cv.undistortPoints(
                    points2, self.K, self.dist, None, self.K)
                # find Essential matrix
                E, _ = cv.findEssentialMat(points1, points2, self.K)
                # decompose E to get pose
                _, R_this, t_this, inlier = cv.recoverPose(
                    E, points1, points2, self.K)
                # Match descriptors in last pairs and current pairs
                matches = bf.match(self.des_last_pair, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                # Get the matched image points and from the matches
                # Get the corresponding points in k-1, k, k+1 view
                # to calculate X of k-1,k view and X of k, k+1 view
                last_X = np.empty((0, 4))
                points_img1 = np.empty((0, 2))
                points_img2 = np.empty((0, 2))
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    last_X = np.vstack((last_X, self.last_X[query_index]))
                    points_img1 = np.vstack((points_img1, points1[img_index]))
                    points_img2 = np.vstack((points_img2, points2[img_index]))
                # reshape
                points_img1 = points_img1.reshape(-1, 1, 2)
                points_img2 = points_img2.reshape(-1, 1, 2)
                # process inlier
                index_inlier = np.argwhere(inlier.flatten()).flatten()
                points1_inlier = points1[index_inlier]
                points2_inlier = points2[index_inlier]
                des2 = des2[index_inlier]
                # form projection matrix of current pair
                t_this = -t_this
                t = self.t + (self.R@t_this)
                R = self.R@R_this
                P = np.hstack((R, t))
                # calculate X
                X = cv.triangulatePoints(self.P, P, points_img1, points_img2)
                X = X/X[-1]
                last_X = last_X.T
                # compute relative scale
                scale = np.mean(np.linalg.norm(
                    X[:, :-1] - X[:, 1:])/np.linalg.norm(last_X[:, :-1] - last_X[:, 1:]))
                # control scale in range of 0 to 3
                scale %= 3
                # compute real pose and form projection matrix
                t = self.t + scale*(self.R@t_this)
                R = self.R@R_this
                P = np.hstack((R, t))
                # record descriptor of current pair
                self.des_last_pair = des2
                # record X of current pair
                self.last_X = cv.triangulatePoints(
                    self.P, P, points1_inlier, points2_inlier)
                self.last_X = (self.last_X/self.last_X[-1]).T
                # record the real pose of current pair
                self.R = R
                self.t = t
                self.P = P
            # save pose
            queue.put((R, t))

            cv.imshow('frame', img)
            if cv.waitKey(30) == 27:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy',
                        help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
