"""
dataset.py 
 - Desc: Dataset Class for Streaming Midi 
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
import json
from tqdm.notebook import tqdm
import pickle 


""" Dataset of Midi Samples 
 - NOTE: Assumes samples generated are in form 
     - sf2_filename
        - Instrument
            - note_vel.wav 
"""
class Midi_Seg(Dataset):
    def __init__(
        self,
        sf_path: str, 
        sr: int = 48000, 
        buffer_size: int = 1024, 
        prev_ratio: float = 2.0):

        super().__init__()
        self.sf_path = sf_path
        self.sr = sr 
        self.buffer_size = buffer_size
        self.prev_ratio = prev_ratio

        self.files = glob.glob(os.path.join(sf_path, '*/*/*.wav'))

    def __getitem__(self, idx):  
        audio_file = self.files[idx]

        x, sr = sf.read(audio_file)

        # Assumes Stereo
        x = torch.from_numpy(x).mean(axis = 1)

        pitch = torch.tensor(int(audio_file.split('/')[-1].split('_')[0]))

        return [x, pitch, self.buffer_size, self.prev_ratio]
    
    def __len__(self): 
        return len(self.files)
    
""" 
train_collate_fn: 
 - Desc: 
"""
def train_collate_fn(batch): 
    x = torch.stack([item[0] for item in batch]) # Combine Audios
    pitch = torch.stack([item[1] for item in batch])

    buffer_size = batch[0][2] # Get Buffer Size 
    prev_ratio = batch[0][3] # Get Previous Info Ratio



    # Where idx sampling can begin and end 
    start_idx = int(buffer_size * prev_ratio)
    end_idx = int(x.shape[-1] - buffer_size)

    # Pad by Prev Ratio
    pad_len = int(buffer_size * prev_ratio)
    x = F.pad(x, pad=(pad_len, 0))

    # For Each Batch, Return 4 Samples (to allow for permutation robusness for timbre embedding)
    rand_idxs = np.random.choice(range(start_idx, end_idx), size = 4)
    xs = []
    prev_xs = []
    for idx in rand_idxs:
        prev_x = x[:, idx - pad_len: idx] # Previous Info
        sub_x = x[:, idx : idx + buffer_size]

        xs.append(sub_x)
        prev_xs.append(prev_x)

    import pdb; pdb.set_trace()

    return xs, prev_xs, pitch 

if __name__ == '__main__': 
    dataset = Midi_Seg('/data/robbizorg/music/samples')
    train_dataloader = DataLoader(dataset, collate_fn=train_collate_fn, batch_size=32, shuffle=True)

    x = dataset.__getitem__(0)
    print(f'Size of Dataset: {len(dataset)}')

    for i, batch in enumerate(train_dataloader):
        break