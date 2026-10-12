"""VLM arm of the OCR bake-off: transcribe the same regions/pages with Bedrock vision models.
usage: run_bedrock.py <bakeoff_base> <repo_root>"""
import json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

base, repo = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from cosorag.llm import LLM  # noqa: E402

PROMPT = ("Transcribe ALL text in this scanned image exactly as printed or handwritten, line by line, in the original "
          "language and script (Hindi in Devanagari, English as is). Keep every number, date, code and punctuation mark. "
          "For tables, write one row per line with cells separated by ' | '. Do not translate, correct, summarize or add "
          "anything. Output only the transcription.")
ENGINES = {
    "vlm-claude-sonnet-4.6": ("bedrock:eu.anthropic.claude-sonnet-4-6", "eu-north-1"),
    "vlm-llama4-maverick": ("bedrock:us.meta.llama4-maverick-17b-instruct-v1:0", "us-east-1"),
    "vlm-gemma3-27b": ("bedrock:google.gemma-3-27b-it", "ap-south-1"),
    "vlm-pixtral-large": ("bedrock:eu.mistral.pixtral-large-2502-v1:0", "eu-north-1"),
}
REGIONS = ["raj1A", "raj1B", "raj2A", "raj2B", "pm10A", "pm20A"]
PAGES = ["raj_p1", "raj_p2", "pmgsy_p10", "pmgsy_p20"]


def run(name, spec, region):
    llm = LLM(spec, region=region)
    out = {"engine": name, "regions": {}, "pages": {}}
    jobs = [("regions", r, base / "regions" / f"{r}.png") for r in REGIONS] + \
           [("pages", p, base / "pages" / f"{p}_300.png") for p in PAGES]
    for kind, key, path in jobs:
        t = time.perf_counter()
        try:
            img = path.read_bytes()
            r = llm.chat("You are a precise OCR engine.", PROMPT, images=[img], max_tokens=4000, temperature=0.0)
            txt = r.text
        except Exception as e:  # noqa: BLE001
            txt = ""
            print(name, key, "ERROR", type(e).__name__, str(e)[:150], flush=True)
        out[kind][key] = {"text": txt, "secs": round(time.perf_counter() - t, 2)}
        print(name, key, f"{time.perf_counter() - t:.1f}s", len(txt), flush=True)
    (base / "out" / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


with ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda kv: run(kv[0], *kv[1]), ENGINES.items()))
