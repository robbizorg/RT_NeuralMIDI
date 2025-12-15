## Reat-Time MusicGen

Real-Time Neural Instrument controllable via Midi Pitch, Velocity, and Instrument Selection

Todos: 
 - [] Figure out Streaming Setup 
 - [X] Create/Collect Dataset (TorchSynth/Carmine's Datasets/Midi Generation)
    - Ended up going with Midi Generation
 - [] Iterate over Architectures, Sort by Runtime


Possible SoundFont Datasets
 - https://huggingface.co/datasets/projectlosangeles/soundfonts4u/tree/main
 - GeneralUser GS (https://schristiancollins.com/generaluser.php)
 - Fluid Release 3 General Midi Soundfont (https://member.keymusician.com/Member/FluidR3_GM/index.html)
 - Musical Artifacts (https://musical-artifacts.com/)
    - Allows musicians to publish and share their various files for music production
    [] Undertale Soundfont (https://musical-artifacts.com/artifacts/914)
    [+] Tohou Soundfont (https://musical-artifacts.com/artifacts/433)
    [] Ultimate MegaDrive Soundfont (https://musical-artifacts.com/artifacts/24)
    [] Roland SC 88 (https://musical-artifacts.com/artifacts/538)
    [+] DSoundFont (https://musical-artifacts.com/artifacts/931); A Massive Soundfont
    [] Alex's GM Soundfont (https://musical-artifacts.com/artifacts/1390)
        - Meant for "Live Sounding" instruments
    [+] Strix's Guitar and Bass Pack (https://musical-artifacts.com/artifacts/1061)

Example Curl Command to Download: `curl -L -o /data/robbizorg/music/soundfiles/Shreddage_II_Revalver_MK_III.V_.sf2 "https://musical-artifacts.com/artifacts/1870/Shreddage_II__Revalver_MK_III.V_.sf2"`

### Requirements 

FluidSynth download: `sudo apt-get install fluidsynth`

### Data Generation: 

Command to Run Generation: `nohup python -u scripts/data_generation/generate_sf_samples.py &>logs/data/data_gen.out &`

Sample Directory Format: 
    - sf2_filename
        - Instrument
            - note_vel.wav


### Training Assumptions

Since we only want to generate sounds that are in the dataset, we are alright with not having a Train-Test Split