"""
Detects sustained frequencies (Hz), durations, and gaps from a monophonic .wav file.

Dependencies:
    pip install numpy librosa soundfile

Usage:
    python beepAnalyser.py beep.wav --out freqs.csv
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import csv

def detect_frequencies(
    filename: str,
    sr: int = 176400, # Sample Rate - Higher SR Improves Time Detail but Results in a Longer Runtime
    fmin: float = 100.0, # Minimum Expected Frequency
    fmax: float = 5000.0, # Maximum Expected Frequency
    frame_length: int = 2048, # Higher Values = Better frequency Resolution but Worse Timing Precision, Lower Values = Better Timing but Less Reliable Frequency
    hop_length: int = 256, # Lower Values = Better Time resolution, Slower Runtime, Jittery Frequency Tracking
    energy_thresh: float = 1e-6,
):
    # Load audio
    y, sr = librosa.load(filename, sr=sr, mono=True)
    duration = len(y) / sr

    # RMS energy for silence detection
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Fundamental frequency estimation
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # Make sure lengths match
    N = min(len(rms), len(f0))
    f0 = f0[:N]
    rms = rms[:N]
    times = times[:N]

    events = []
    cur_state = None  # "note" or "rest"
    cur_freq = None
    start_t = 0.0

    def flush_segment(end_t):
        nonlocal cur_state, cur_freq, start_t
        if cur_state is None:
            return
        dur = end_t - start_t
        if dur <= 0:
            return
        if cur_state == "note":
            events.append({
                "type": "note",
                "freq_hz": cur_freq,
                "start_s": start_t,
                "duration_s": dur
            })
        else:
            events.append({
                "type": "rest",
                "freq_hz": None,
                "start_s": start_t,
                "duration_s": dur
            })
        cur_state = None
        cur_freq = None
        start_t = end_t

    # Iterate frames
    for i in range(N):
        is_voiced = not np.isnan(f0[i]) and rms[i] > energy_thresh
        if is_voiced:
            freq = float(f0[i])
            if cur_state != "note":
                flush_segment(times[i])
                cur_state = "note"
                cur_freq = freq
                start_t = times[i]
            else:
                # Smooth frequency (running average)
                cur_freq = 0.9 * cur_freq + 0.1 * freq
        else:
            if cur_state != "rest":
                flush_segment(times[i])
                cur_state = "rest"
                start_t = times[i]

    flush_segment(duration)

    # Compute gaps: duration of following rest
    for idx, ev in enumerate(events):
        gap_after = 0.0
        if ev["type"] == "note" and idx + 1 < len(events) and events[idx + 1]["type"] == "rest":
            gap_after = events[idx + 1]["duration_s"]
        ev["gap_after_s"] = gap_after

    return events, duration


def print_events(events):

    for ev in events:
        frequency = ev['freq_hz']
        duration = ev['duration_s'] * 1000
        gap = ev['gap_after_s'] * 1000

        if ev["type"] == "note":
            print(f"beep({int(frequency)}, {int(duration)});\ndelay({int(gap)});")


def save_csv(events, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "freq_hz", "start_s", "duration_s", "gap_after_s"])
        writer.writeheader()
        for ev in events:
            writer.writerow({
                "type": ev["type"],
                "freq_hz": f"{ev['freq_hz']:.2f}" if ev["freq_hz"] else "",
                "start_s": f"{ev['start_s']:.6f}",
                "duration_s": f"{ev['duration_s']:.6f}",
                "gap_after_s": f"{ev['gap_after_s']:.6f}"
            })


def main():
    parser = argparse.ArgumentParser(description="Detect frequencies, durations and gaps from a WAV file.")
    parser.add_argument("infile", help="Input .wav file")
    parser.add_argument("--out", help="Optional CSV output path", default=None)
    args = parser.parse_args()

    events, total_dur = detect_frequencies(args.infile)
    print(f"\nFile: {args.infile}  (duration {total_dur:.2f}s)\n")
    print_events(events)

    if args.out:
        save_csv(events, args.out)
        print(f"\nSaved CSV to {args.out}")


if __name__ == "__main__":
    main()