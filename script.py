#!/usr/bin/env python3
"""Your own song in one command: spoken lyrics + a cappella singing + the lyrics -> you singing them.

    python3 script.py --speech sample/speech.mp3 --melody sample/melody.mp3 \\
                      --lyrics sample/lyrics.txt --out outputs/custom/my_song.wav

1. aligns the lyrics to both recordings with the Montreal Forced Aligner (english_us_arpa),
2. writes the alignments as NUS-48E phone annotations next to --out (<stem>_speech.txt, <stem>_melody.txt),
3. runs infer.py on them with --postfilter sharpen_gv; any other infer.py option is passed
   through (e.g. --transpose -12, --durations phrase, --postfilter none).

English lyrics only; the singing must be vocals only. MFA breaks on paths with spaces, so it
lives outside the project (~/mfa-env) and runs in a temporary directory:
    micromamba create -p ~/mfa-env -c conda-forge montreal-forced-aligner   (or conda create -p ...)
plus english_us_arpa.dict and english_us_arpa_acoustic.zip in ~/mfa-env/models/
(https://github.com/MontrealCorpusTools/mfa-models/releases).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import infer  # noqa: E402
from features.audio import load_numpy  # noqa: E402

MFA_SR = 16000
DICT, ACOUSTIC = "english_us_arpa.dict", "english_us_arpa_acoustic.zip"


def to_phones(mfa_json, duration, path):
    """MFA phone tier -> NUS-48E annotation: lowercase ARPAbet without stress, every gap as sil."""
    rows, t = [], 0.0
    for s, e, ph in json.load(open(mfa_json))["tiers"]["phones"]["entries"]:
        ph = re.sub(r"\d", "", ph).lower()
        if ph in ("", "sil", "sp", "spn"):
            continue
        if s - t > 1e-4:
            rows.append((t, s, "sil"))
        rows.append((s, e, ph))
        t = e
    if duration - t > 1e-4:
        rows.append((t, duration, "sil"))
    Path(path).write_text("".join(f"{s:.6f} {e:.6f} {p}\n" for s, e, p in rows))


def align(recordings: dict, lyrics: str, mfa: str, models: Path, out: Path) -> dict:
    """{name: audio path} -> {name: phone annotation path} written next to out."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp, durations = Path(tmp), {}
        for name, audio in recordings.items():
            (tmp / "corpus" / name).mkdir(parents=True)  # one MFA speaker per recording
            wav = load_numpy(audio, MFA_SR)
            sf.write(tmp / "corpus" / name / f"{name}.wav", wav, MFA_SR)
            (tmp / "corpus" / name / f"{name}.lab").write_text(lyrics)
            durations[name] = len(wav) / MFA_SR
        for f in (DICT, ACOUSTIC):
            shutil.copy(models / f, tmp)
        env = {**os.environ, "MFA_ROOT_DIR": str(tmp / "root"),
               "PATH": str(Path(mfa).parent) + os.pathsep + os.environ["PATH"]}
        run = subprocess.run([mfa, "align", "corpus", DICT, ACOUSTIC, "aligned", "--clean", "--beam", "100",
                              "--retry_beam", "400", "--output_format", "json", "-j", "4"],
                             cwd=tmp, env=env, capture_output=True, text=True)
        phones = {}
        for name in recordings:
            result = tmp / "aligned" / name / f"{name}.json"
            if run.returncode or not result.exists():
                sys.exit(f"MFA could not align {recordings[name]}:\n" + (run.stdout + run.stderr)[-3000:])
            phones[name] = out.with_name(f"{out.stem}_{name}.txt")
            to_phones(result, durations[name], phones[name])
    return phones


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="Other options (--transpose, --durations, --postfilter, ...) go to infer.py.")
    ap.add_argument("--speech", required=True, help="you speaking the lyrics (any format soundfile reads)")
    ap.add_argument("--melody", required=True, help="a cappella singing of the same lyrics")
    ap.add_argument("--lyrics", required=True, help="the lyrics (English text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mfa", default=str(Path.home() / "mfa-env" / "bin" / "mfa"))
    ap.add_argument("--mfa-models", default=str(Path.home() / "mfa-env" / "models"))
    args, rest = ap.parse_known_args()

    models, lyrics = Path(args.mfa_models), Path(args.lyrics).read_text()
    for path in (Path(args.mfa), models / DICT, models / ACOUSTIC):
        if not path.exists():
            sys.exit(f"missing {path}: see the setup in the header of script.py")
    vocab = {line.split("\t", 1)[0] for line in open(models / DICT, encoding="utf-8")}
    unknown = sorted(set(re.findall(r"[a-z0-9']+", lyrics.lower())) - vocab)
    if unknown:  # MFA would align them as noise, losing those words
        sys.exit(f"not in the pronunciation dictionary: {' '.join(unknown)}; "
                 f"respell them as they sound in {args.lyrics}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("aligning the lyrics with MFA ...", flush=True)
    phones = align({"speech": args.speech, "melody": args.melody}, lyrics, args.mfa, models, out)
    if not any(a.startswith("--postfilter") for a in rest):
        rest += ["--postfilter", "sharpen_gv"]
    infer.main(["--speech", args.speech, "--speech-phones", str(phones["speech"]), "--melody", args.melody,
                "--melody-phones", str(phones["melody"]), "--out", str(out), *rest])


if __name__ == "__main__":
    main()
