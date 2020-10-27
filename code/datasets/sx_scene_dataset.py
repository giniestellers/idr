import os
import torch
import numpy as np

import glob
import json
import imageio
import skimage
import cv2
import transformations

import utils.general as utils

class SxSceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras=False,
                 basedir='',
                 downscale_factor=1
                 ):

        self.instance_dir = basedir
        self.scale_matrix = None
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.img_res = None
        self.train_cameras = train_cameras

        self.rgb_images = []
        self.object_masks = []
        self.intrinsics_all = []
        self.pose_all = []
        metadata_file_paths = glob.glob(os.path.join(basedir, "meta_*.json"))
        for json_path in metadata_file_paths:
            with open(json_path, "r") as fp:
                metadata = json.load(fp)

            json_filename = os.path.splitext(os.path.basename(json_path))[0]
            file_suffix = "_".join(json_filename.split("_")[1:])
            img_path = os.path.join(basedir, "img_" + file_suffix + ".png")
            H, W = 1, 1
            mask_paths = []
            mask_paths.append(os.path.join(basedir, "segm_face_" + file_suffix + ".png"))
            mask_paths.append(os.path.join(basedir, "segm_top_empty_face_" + file_suffix + ".png"))
            mask_paths.append(os.path.join(basedir, "segm_headwear_empty_face_" + file_suffix + ".png"))
            mask_paths.append(os.path.join(basedir, "segm_glasses_face_" + file_suffix + ".png"))

            # load image
            assert os.path.exists(img_path), "missing rgb image"
            img = imageio.imread(img_path)
            img_shape = img.shape[:2]
            if downscale_factor != 1:
                H = img.shape[0] // 2**(downscale_factor - 1)
                W = img.shape[1] // 2**(downscale_factor - 1)
                downscaled_img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
                img = downscaled_img
            img_res = img.shape[:2]
            if self.img_res == None:
                self.img_res = img_res
            else:
                assert img_res == self.img_res, "image sizes are not consistent accross cameras"
            # pixel values between [-1,1]
            img = skimage.img_as_float32(img)
            img -= 0.5
            img *= 2.
            img = img.reshape(-1, 3)
            self.rgb_images.append(torch.from_numpy(img).float())

            # load mask
            combined_mask = np.zeros(img_shape, dtype=np.int32)
            for mask_path in mask_paths:
                if os.path.exists(mask_path):
                    combined_mask += imageio.imread(mask_path)
            combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
            if downscale_factor != 1:
                downscaled_mask = cv2.resize(combined_mask, (H, W), interpolation=cv2.INTER_AREA)
                combined_mask = downscaled_mask
            assert combined_mask.shape[:2] == self.img_res, "size of image and mask inconsistent"
            combined_mask = combined_mask > 127.5
            combined_mask = combined_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(combined_mask).bool())

            # load camera pose
            camera_location = np.reshape(metadata["cameras"][0]["location"], (3, 1))
            camera_rotation = metadata["cameras"][0]["rotation"]
            rotation_matrix = transformations.euler_matrix(*camera_rotation)[:3, :3]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :] = np.hstack((rotation_matrix, camera_location))
            # might need to scale pose to unit sphere
            self.pose_all.append(torch.from_numpy(pose))

            # load camera parameters
            camera_focal_mm = metadata["cameras"][0]["fov"] / 100
            frame_width_mm = 36
            frame_width_pix = self.img_res[1]
            focal = frame_width_pix * (camera_focal_mm / frame_width_mm)
            intrinsics = np.array([[float(focal), 0.0, 0.5*self.img_res[1]],
                            [0, float(focal), 0.5*self.img_res[0]],
                            [0, 0, 1]], dtype=np.float32)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())

        self.n_images = len(self.rgb_images)
        self.total_pixels = self.img_res[0]*self.img_res[1]
        self.scale_coordinate_frame()

    def scale_coordinate_frame(self):
        if self.scale_matrix is None:
            self.scale_matrix = self.get_scale_mat()
            for i in range(self.n_images):
                pose = self.pose_all[i] @ self.scale_matrix
                self.pose_all[i] = pose

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        if self.scale_matrix is None:
            camera_centers = np.stack(self.pose_all, 0)[:, :3, 3]
            rig_center = np.mean(camera_centers, 0, keepdims=True)
            centered_cameras = camera_centers - np.repeat(rig_center, self.n_images, axis=0)
            scale = np.max(np.linalg.norm(centered_cameras, ord=2, axis=-1))
            self.scale_matrix = scale * np.eye(4)
            self.scale_matrix[:3, 3] = rig_center[0]
        return self.scale_matrix

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        if scaled:
            pose_all = list(self.pose_all)
        else:
            inverse_scale = 1.0/self.scale_matrix[0,0]
            inverse_scale_matrix = inverse_scale * np.eye(4)
            inverse_scale_matrix[:3, 3] = - inverse_scale * self.scale_matrix[:3, 3]
            pose_all = [ pose @ inverse_scale_matrix for pose in self.pose_all]
        return pose_all

    def get_pose_init(self):
        return list(self.pose_all)


if __name__ == "__main__":
    pass