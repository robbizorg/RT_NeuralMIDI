"""
utils.py 
 - Desc: Contains SoundFile utils for loading and generating audio from soundfiles
"""

import numpy as np 
import soundfile as sf 
from sf2utils.sf2parse import Sf2File
import os 

import logging

# Suppress sf2utils warnings about midi start and stops
# Warning (Ironic): Suppresses all warnings
logging.getLogger().setLevel(logging.ERROR)

# Midi Imports
import tempfile
import pretty_midi
import fluidsynth



def midi_to_note_name(n):
    """Converts MIDI note number → name like C4."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (n // 12) - 1
    name = names[n % 12]
    return f"{name}{octave}"

def inspect_soundfont(sf2_path: str):
    # Load SoundFont
    with open(sf2_path, "rb") as f:
        sf2 = Sf2File(f)

    print(f"Loaded SoundFont: {sf2_path}")
    print("=" * 80)

    for preset in sf2.presets:
        # Skip sentinel preset (EOP = End Of Presets)
        if getattr(preset, "name", None) == "EOP":
            continue

        print(f"\nPreset: {preset.name}  (Bank {preset.bank}, Program {preset.preset})")
        print("-" * 80)

        # To avoid printing exact duplicates
        seen = set()

        # Preset-level bags (can include a "global" bag with no instrument)
        for pbag in preset.bags:
            instrument = pbag.instrument
            if instrument is None:
                # Global bag: carries default gens like global key_range, etc.
                continue

            preset_key_range = pbag.key_range  # may be None

            # Instrument-level bags: usually where samples + key ranges live
            for ibag in instrument.bags:
                sample = ibag.sample
                if sample is None:
                    continue  # e.g., global instrument bag

                # Determine effective key range:
                # 1) instrument bag key_range if present
                # 2) else preset bag key_range
                # 3) else full MIDI range
                if ibag.key_range is not None:
                    lo, hi = ibag.key_range
                elif preset_key_range is not None:
                    lo, hi = preset_key_range
                else:
                    lo, hi = 0, 127

                # Determine root key:
                # - bag.base_note (overriding root key) if present
                # - else sample.original_pitch
                base_note = getattr(ibag, "base_note", None)
                if base_note is not None:
                    root_key = base_note
                else:
                    root_key = sample.original_pitch

                sig = (instrument.name, sample.name, root_key, lo, hi)
                if sig in seen:
                    continue
                seen.add(sig)

                print(f"  Instrument: {instrument.name}")
                print(f"    Sample: {sample.name}")
                print(f"      Root key: {root_key} ({midi_to_note_name(root_key)})")
                print(
                    f"      Key range: {lo}–{hi} "
                    f"({midi_to_note_name(lo)} → {midi_to_note_name(hi)})"
                )

# Render a single note a numpy array
def render_note_to_numpy(
    sf2_path: str,
    midi_pitch: int = 60,      # C4
    velocity: int = 100,
    duration_sec: float = 2.0,
    program: int = 0,          # GM program number
    sample_rate: int = 44100,
):
    """
    Render a single MIDI note from a SoundFont to a NumPy array using FluidSynth.
    Returns: np.ndarray shape (num_samples, 2) for stereo audio.
    """
    # Create synthesizer
    fs = fluidsynth.Synth(samplerate=sample_rate)
    sfid = fs.sfload(sf2_path)
    fs.program_select(0, sfid, 0, program)

    # Start note
    fs.noteon(0, midi_pitch, velocity)

    # Render audio into buffer
    num_frames = int(duration_sec * sample_rate)
    audio = fs.get_samples(num_frames)  # returns float32 array interleaved L/R

    # Stop note
    fs.noteoff(0, midi_pitch)

    # Clean up
    fs.delete()

    # Convert interleaved stereo → shape (N, 2)
    audio = np.array(audio, dtype=np.float32)
    audio = audio.reshape(-1, 2)  # stereo

    return audio

def get_soundfont_structure(sf2_path):
    with open(sf2_path, "rb") as f:
        sf2 = Sf2File(f)

    presets = []

    for preset in sf2.presets:
        if preset.name == "EOP":     # Skip sentinel ending preset
            continue
        presets.append({
            "bank": preset.bank,
            "program": preset.preset,
            "name": preset.name,
            "preset_obj": preset,
        })

    return presets