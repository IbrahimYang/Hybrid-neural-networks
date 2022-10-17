import os
import numpy as np
import cv2
from dv import AedatFile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class TurningDiskDataset(Dataset):
    def __init__(self, test=False) -> None:
        super().__init__()
        if not test:
            self.events = np.load(abspath("aedat4_data/dvSave-2020_07_23_10_28_47_event.npy"))
            self.frames = np.load(abspath("aedat4_data/dvSave-2020_07_23_10_28_47_frame.npy"))
        else:
            self.events = np.load(abspath("aedat4_data/dvSave-2020_07_23_10_28_03_event.npy"))
            self.frames = np.load(abspath("aedat4_data/dvSave-2020_07_23_10_28_03_frame.npy"))

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, item):
        aps = self.frames[item]
        dvs = self.events[item]
        aps_loc = self.get_target(aps.transpose(1, 2, 0))
        dvs_loc = np.stack([self.get_target(dvs[1:, ..., t].transpose(1, 2, 0), isFrame=False) for t in range(dvs.shape[-1])], -1)
        return aps, dvs, aps_loc, dvs_loc

    @staticmethod
    def get_target(img, isFrame=True):
        if isFrame:
            img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        else:
            img = (img > 0).astype(np.uint8)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_sort = np.argsort([cv2.contourArea(c) for c in contours])
        area_sort = area_sort[-2:-5:-1] if isFrame else area_sort[:-4:-1]
        contours = np.asarray(contours, dtype=object)[area_sort]

        x, y, w, h = np.array([cv2.boundingRect(cnt) for cnt in contours]).T
        xc, yc = x + w / 2, y + h / 2
        center = np.stack([xc, yc], -1)
        r = ((img.shape[1] / 2 - xc) ** 2 + (img.shape[0] / 2 - yc) ** 2) ** 0.5
        center = center[np.argsort(r)]

        center = center - np.array(img.shape[:2])[np.newaxis, ::-1] / 2 + 0.5
        center = center / 8
        return center


def aedat4_2_numpy(file_path, T_event_frame=1, event_frame_num=5*2):
    file_name_list = []
    for file in os.listdir(file_path):
        if "aedat4" in file:
            file_name_list.append(file)
    file_name_list.sort()

    for file_name in file_name_list:
        print("processing "+file_name+" ...")
        file = AedatFile(abspath("aedat4_data/"+file_name))
        event = next(file['events'])
        start_frame_time = next(file['frames']).timestamp
        blank_img = np.zeros((2,)+file['events'].size+(event_frame_num,), dtype=np.uint8)
        frames_data, events_data = [], []

        for frame in file['frames']:
            print("frame time: {:.3f}ms".format((frame.timestamp - start_frame_time)/1e3))
            img = cv2.resize(frame.image, (145, 145))[np.newaxis, ...]
            frames_data.append(img)
            frame_time = frame.timestamp - T_event_frame * event_frame_num / 2 * 1e3
            c = 0
            event_img = blank_img.copy()
            while event.timestamp < frame_time:
                event = next(file['events'])
            while event.timestamp < frame.timestamp + T_event_frame * event_frame_num / 2 * 1e3:
                while event.timestamp - frame_time < T_event_frame * 1e3:
                    if event.polarity:
                        event_img[0, event.y, event.x, c] += 1
                    else:
                        event_img[1, event.y, event.x, c] += 1
                    event = next(file['events'])
                frame_time += T_event_frame * 1e3
                c += 1
            cropped_img = event_img[:, 71:71+145, 88:88+145, :]
            events_data.append(cropped_img)

        np.save(abspath("aedat4_data/"+file_name[:file_name.find('.')]+"_frame.npy"), np.array(frames_data, dtype=np.float32))
        np.save(abspath("aedat4_data/"+file_name[:file_name.find('.')]+"_event.npy"), np.array(events_data, dtype=np.float32))
        file.close()


def abspath(path):
    return os.path.join(os.path.dirname(__file__), path)


if __name__ == "__main__":
    aedat4_2_numpy(abspath("aedat4_data"))
    # data = TurningDiskDataset()[1]
