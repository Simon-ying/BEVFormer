from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import sys
import os
import mmengine
sys.path.append(".")
from nuscenes.utils import splits
import numpy as np


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

version = "v1.0-mini"
root_path = "./data/nuscenes"
can_bus_root_path = "./data"

nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)

available_scenes = get_available_scenes(nusc)
available_scene_names = [s['name'] for s in available_scenes]
print(f"available scenes: {available_scene_names}")

train_scenes = splits.mini_train
val_scenes = splits.mini_val

print(f"train scenes: {train_scenes}")
print(f"val scenes: {val_scenes}")

train_scenes = list(
    filter(lambda x: x in available_scene_names, train_scenes))
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

print(f"train scenes: {train_scenes}")
print(f"val scenes: {val_scenes}")

# get scene tokens for each scene
train_scenes = set([
    available_scenes[available_scene_names.index(s)]['token']
    for s in train_scenes
])
val_scenes = set([
    available_scenes[available_scene_names.index(s)]['token']
    for s in val_scenes
])

print(f"train scenes: {train_scenes}")
print(f"val scenes: {val_scenes}")

sample = nusc.sample[0]
lidar_token = sample['data']['LIDAR_TOP']
sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
cs_record = nusc.get('calibrated_sensor',
                        sd_rec['calibrated_sensor_token'])
pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

scene_name = nusc.get('scene', sample['scene_token'])['name']
sample_timestamp = sample['timestamp']
pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
can_bus = []
# during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
last_pose = pose_list[0]
for i, pose in enumerate(pose_list):
    if pose['utime'] > sample_timestamp:
        break
    last_pose = pose
_ = last_pose.pop('utime')  # useless
pos = last_pose.pop('pos')
rotation = last_pose.pop('orientation')
can_bus.extend(pos)
can_bus.extend(rotation)
for key in last_pose.keys():
    can_bus.extend(pose[key])  # 16 elements
can_bus.extend([0., 0.])
print(f"last pose: {last_pose}, can bus:{can_bus}")
print(cs_record['rotation'])

locs = np.array([b.center for b in boxes]).reshape(-1, 3)
dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
rots = np.array([b.orientation.yaw_pitch_roll[0]
                    for b in boxes]).reshape(-1, 1)
print(rots.shape)