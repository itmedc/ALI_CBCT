import GlobalVar as GV

import SimpleITK as sitk
import numpy as np
import torch
import os
import json

from monai.transforms import (
    Compose,
    ScaleIntensity,
    SpatialCrop,
    BorderPad,
)

from utils import(
    ReadFCSV,
    SetSpacing,
    ItkToSitk,
    GenControlePoint,
    WriteJson
)

# #####################################
#  Environment
# #####################################

class Environment :
    def __init__(
        self,
        patient_id,
        padding,
        device,
        correct_contrast = False,
        verbose = False,

    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.patient_id = patient_id
        self.padding = padding.astype(np.int16)
        self.device = device
        self.verbose = verbose
        add_channel = lambda data: data[None]
        self.transform = Compose([add_channel, BorderPad(spatial_border=self.padding.tolist())])

        self.scale_nbr = 0

        self.available_lm = []

        self.data = {}

        self.predicted_landmarks = {}


    def LoadImages(self,images_path):

        scales = []

        for scale_id,path in images_path.items():
            data = {"path":path}
            img = sitk.ReadImage(path)
            img_ar = sitk.GetArrayFromImage(img)
            data["image"] = torch.as_tensor(self.transform(img_ar)).type(torch.int16)

            data["spacing"] = np.array(img.GetSpacing())
            origin = img.GetOrigin()
            data["origin"] = np.array([origin[2],origin[1],origin[0]])
            data["size"] = np.array(np.shape(img_ar))

            data["landmarks"] = {}

            self.data[scale_id] = data
            self.scale_nbr += 1

            

    def LoadJsonLandmarks(self,fiducial_path):

        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            if markup["label"] not in GV.LABELS:
                print(fiducial_path)
                print(f"{GV.bcolors.WARNING}WARNING : {markup['label']} is an unusual landmark{GV.bcolors.ENDC}")
            mark_pos = markup["position"]
            lm_ph_coord = np.array([mark_pos[2],mark_pos[1],mark_pos[0]])
            self.available_lm.append(markup["label"])
            for scale,scale_data in self.data.items():
                lm_coord = ((lm_ph_coord - scale_data["origin"]) / scale_data["spacing"]).astype(np.int16)
                scale_data["landmarks"][markup["label"]] = lm_coord


    def SavePredictedLandmarks(self,scale_key,out_path=None):
        img_path = self.data[scale_key]["path"]
        print(f"Saving predicted landmarks for patient{self.patient_id} at scale {scale_key}")

        ref_origin = self.data[scale_key]["origin"]
        ref_spacing = self.data[scale_key]["spacing"]
        physical_origin = -ref_origin / ref_spacing

        landmark_dic = {}
        for landmark,pos in self.predicted_landmarks.items():

            real_label_pos = (pos-physical_origin)*ref_spacing
            real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
            if GV.LABEL_GROUPES[landmark] in landmark_dic.keys():
                landmark_dic[GV.LABEL_GROUPES[landmark]].append({"label": landmark, "coord":real_label_pos})
            else:landmark_dic[GV.LABEL_GROUPES[landmark]] = [{"label": landmark, "coord":real_label_pos}]


        for group,list in landmark_dic.items():

            id = self.patient_id.split(".")[0]
            json_name = f"{id}_lm_Pred_{group}.mrk.json"

            if out_path is not None:
                file_path = os.path.join(out_path,json_name)
            else:
                file_path = os.path.join(os.path.dirname(img_path),json_name)
            groupe_data = {}
            for lm in list:
                groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

            lm_lst = GenControlePoint(groupe_data)
            WriteJson(lm_lst,file_path)

    def ResetLandmarks(self):
        for scale in self.data.keys():
            self.data[scale]["landmarks"] = {}

        self.available_lm = []

    def LandmarkIsPresent(self,landmark):
        return landmark in self.available_lm

    def GetLandmarkPos(self,scale,landmark):
        return self.data[scale]["landmarks"][landmark]

    def GetL2DistFromLandmark(self, scale, position, target):
        label_pos = self.GetLandmarkPos(scale,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,scale,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        rescale = ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)
        crop = cropTransform(self.data[scale]["image"])
        crop = rescale(crop).type(torch.float32)
        return crop

    def GetRewardLst(self,scale,position,target,mvt_matrix):
        agent_dist = self.GetL2DistFromLandmark(scale,position,target)
        get_reward = lambda move : agent_dist - self.GetL2DistFromLandmark(scale,position + move,target)
        reward_lst = list(map(get_reward,mvt_matrix))
        return reward_lst
    
    def GetRandomPoses(self,scale,target,radius,pos_nbr):
        if scale == GV.SCALE_KEYS[0]:
            percentage = 0.2
            centered_pos_nbr = int(percentage*pos_nbr)
            rand_coord_lst = self.GetRandomPosesInAllScan(scale,pos_nbr-centered_pos_nbr)
            rand_coord_lst += self.GetRandomPosesAroundLabel(scale,target,radius,centered_pos_nbr)
        else:
            rand_coord_lst = self.GetRandomPosesAroundLabel(scale,target,radius,pos_nbr)

        return rand_coord_lst

    def GetRandomPosesInAllScan(self,scale,pos_nbr):
        max_coord = self.data[scale]["size"]
        get_rand_coord = lambda x: np.random.randint(1, max_coord, dtype=np.int16)
        rand_coord_lst = list(map(get_rand_coord,range(pos_nbr)))
        return rand_coord_lst
    
    def GetRandomPosesAroundLabel(self,scale,target,radius,pos_nbr):
        min_coord = [0,0,0]
        max_coord = self.data[scale]["size"]
        landmark_pos = self.GetLandmarkPos(scale,target)

        get_random_coord = lambda x: landmark_pos + np.random.randint([1,1,1], radius*2) - radius

        rand_coords = map(get_random_coord,range(pos_nbr))

        correct_coord = lambda coord: np.array([min(max(coord[0],min_coord[0]),max_coord[0]),min(max(coord[1],min_coord[1]),max_coord[1]),min(max(coord[2],min_coord[2]),max_coord[2])])
        rand_coords = list(map(correct_coord,rand_coords))

        return rand_coords

    def GetSampleFromPoses(self,scale,target,pos_lst,crop_size,mvt_matrix):

        get_sample = lambda coord : {
            "state":self.GetZone(scale,coord,crop_size),
            "target": np.argmax(self.GetRewardLst(scale,coord,target,mvt_matrix))
            }
        sample_lst = list(map(get_sample,pos_lst))

        return sample_lst

    def GetSpacing(self,scale):
        return self.data[scale]["spacing"]

    def GetSize(self,scale):
        return self.data[scale]["size"]

    def AddPredictedLandmark(self,lm_id,lm_pos):
        self.predicted_landmarks[lm_id] = lm_pos

    def __str__(self):
        print(self.patient_id)
        for scale in self.data.keys():
            print(f"{scale}")
            print(self.data[scale]["spacing"])
            print(self.data[scale]["origin"])
            print(self.data[scale]["size"])
            print(self.data[scale]["landmarks"])
        return ""
