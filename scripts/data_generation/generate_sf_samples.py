"""
generate_sf_samples.py 
 - Desc: Given a list of soundfiles, loop through the files and pitches to generate 4s long samples of the instrument
"""

import numpy as np 
import soundfile as sf 
from sf2utils.sf2parse import Sf2File
import os 
import glob 
import sys 
import logging

# Suppress sf2utils warnings about midi start and stops
# Warning (Ironic): Suppresses all warnings
logging.getLogger().setLevel(logging.ERROR)

# Midi Imports
import tempfile
import pretty_midi
import fluidsynth

from dotenv import load_dotenv

load_dotenv()

music_path = os.getenv("music_path")
sf_path = os.getenv('sf_path')
sample_path = os.getenv('sample_path')

# Add Module Imports
sys.path.append('./')
from src.utils import *

if __name__ == '__main__': 
    # Will use standard piano pitch range
    start_p = 21
    end_p = 109
    pitch_range = np.arange(start_p, end_p, dtype=int)

    sr = 48000
    print(f'Generating Audio at {sr}Hz')

    # soundfiles = glob.glob(os.path.join(sf_path, '*.sf2'))
    soundfiles = []
    soundfiles.append(os.path.join(sf_path, 'Touhou.sf2'))
    soundfiles.append(os.path.join(sf_path, 'DSoundfont_Ultimate.sf2'))
    soundfiles.append(os.path.join(sf_path, 'Shreddage_II_Revalver_MK_III.V_.sf2'))

    remake = False
    for soundfile in soundfiles: 
        print(f'Processing {soundfile}')

        sf_name = soundfile.split('/')[-1].split('.')[0]

        print(f'Making dir {sf_name}')

        sf_dir = os.path.join(sample_path, sf_name)
        if not os.path.exists(sf_dir): 
            os.mkdir(sf_dir)

        # Get Presets from soundfile 
        presets = get_soundfont_structure(soundfile)

        print(f'Number of Presets to Process: {len(presets)}')

        for p in presets: 
            bank = p["bank"]
            program = p["program"]
            name = p["name"]

            inst_name = name.replace('_', '-').replace(' ', '-')
            inst_path = os.path.join(sf_dir, inst_name)
            try:
                if not os.path.exists(inst_path): 
                    os.mkdir(inst_path)
            except: 
                print(f'Failed on {inst_path}, continuing')
                continue

            print(f'Processing {inst_path}')

            for midi_p in pitch_range: 
                filename = f'{midi_p}_80.wav'
                audio_path = os.path.join(inst_path, filename)

                if not remake and os.path.exists(audio_path): 
                    continue

                audio = render_note_to_numpy(
                    sf2_path=soundfile,
                    midi_pitch=midi_p,
                    velocity=80,
                    duration_sec=4.0,
                    program=program,
                    sample_rate=sr
                ) # [sr*duration, C]



                # Normalize Audio Before Writing
                peak = np.max(np.abs(audio)) if audio.size else 0.0

                if peak > 0:
                    audio = audio / peak * 0.9  # normalize to -1..1 range with a bit of headroom

                sf.write(audio_path, audio, samplerate=sr)

