"""Transcribe Saathiya's vocal stem with word timestamps, mapped to the 88 BPM bar grid."""
import json, os, sys
import mlx_whisper

OUT_JSON = os.path.expanduser("~/Documents/dnb-remix/output/long/lyrics.json")
STEM = os.path.expanduser("~/Documents/dnb-remix/stems/htdemucs_ft/saathiya/vocals.wav")
BAR = 4 * 60 / 88
G0 = 306.772 - 112 * BAR
model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-large-v3-turbo"

res = mlx_whisper.transcribe(STEM, path_or_hf_repo=model, language="hi",
                             word_timestamps=True, condition_on_previous_text=False,
                             initial_prompt="साथिया, हँसते रहना")
json.dump(res, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)


def pos(t):
    b = (t - G0) / BAR
    return f"bar {int(b):3d} beat {(b % 1) * 4 + 1:4.2f}"


for seg in res["segments"]:
    words = seg.get("words", [])
    print(f"\n[{seg['start']:6.1f}-{seg['end']:6.1f}] {pos(seg['start'])}  {seg['text'].strip()}")
    for w in words:
        print(f"    {w['start']:7.2f} {pos(w['start'])}  {w['word'].strip()}  ({w.get('probability', 0):.2f})")
