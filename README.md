## Real-Time MusicGen

Real-Time Neural Instrument controllable via Midi Pitch, Velocity, and Instrument Selection

Used SoundFont Datasets
 - Tohou Soundfont (https://musical-artifacts.com/artifacts/433)
 - DSoundFont (https://musical-artifacts.com/artifacts/931); A Massive Soundfont

Example Curl Command to Download: `curl -L -o ./samples/Shreddage_II_Revalver_MK_III.V_.sf2 "https://musical-artifacts.com/artifacts/1870/Shreddage_II__Revalver_MK_III.V_.sf2"`

### Requirements 

For MIDI dataset generation, download FluidSynth: `sudo apt-get install fluidsynth`

Then run `pip install -r requirements.txt`

### Data Generation: 

To download and generate your own MIDI Dataset: 

`mkdir dataset`

Then, please create a .env file with the following path variables declared: 
 - music_path='./dataset/music'
 - sf_path='./dataset/music/soundfiles'
 - sample_path='./dataset/music/samples'

Command to Run Generation: `nohup python -u scripts/data_generation/generate_sf_samples.py &>logs/data/data_gen.out &`

Sample Directory Format: 
    - sf2_filename
        - Instrument
            - note_vel.wav


### Training Instructions

Example Command: `CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.main &>logs/training/midi_vocos_1st.out &`