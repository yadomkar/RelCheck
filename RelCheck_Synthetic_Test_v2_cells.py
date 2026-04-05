# ============================================================
# RelCheck v2 — Synthetic Hallucination Test
# ============================================================
# Copy-paste each cell block into Google Colab.
# Cells are separated by: # ── CELL N ──


# ── CELL 1 — Setup ──────────────────────────────────────────
# !pip install together Pillow requests transformers rapidfuzz spacy tenacity json-repair pysbd nltk -q
# !python -m spacy download en_core_web_sm -q

import os, json, time, random
from pathlib import Path
from collections import Counter
from PIL import Image
from google.colab import drive

# ── Config ──
TOGETHER_API_KEY = ''   # <-- paste your key
N_IMAGES         = 20
CAPTIONER        = 'llava'   # 'blip2' | 'llava' | 'qwen'
RANDOM_SEED      = 42
SAVE_DIR         = '/content/drive/MyDrive/RelCheck_Data/synthetic_test'

drive.mount('/content/drive')
os.makedirs(SAVE_DIR, exist_ok=True)
os.environ['TOGETHER_API_KEY'] = TOGETHER_API_KEY

# ── Ensure nltk data ──
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Clone repo + add to path ──
import sys
REPO_DIR = '/content/RelCheck'
if not os.path.exists(os.path.join(REPO_DIR, '.git')):
    os.system(f'git clone https://github.com/yadomkar/RelCheck.git {REPO_DIR}')
else:
    os.system(f'cd {REPO_DIR} && git pull -q')
sys.path.insert(0, REPO_DIR)

# ── Init relcheck_v2 ──
from relcheck_v2.api import init_client
from relcheck_v2.models import get_gdino, get_llava, get_blip2, DEVICE
from relcheck_v2.detection import detect_objects
from relcheck_v2.verification import verify_triple, classify_relation
from relcheck_v2.kb import build_visual_kb
from relcheck_v2.captioning import caption_image
from relcheck_v2.injection import question_to_statement, parse_question, classify_rel_type
from relcheck_v2.evaluation import run_synthetic_rpope
from relcheck_v2.correction import correct_long_caption
from relcheck_v2.data import load_rbench, load_images

init_client(TOGETHER_API_KEY)

# ── Load models ──
import torch
print(f'Device: {DEVICE}')
gdino_model, gdino_processor = get_gdino()
if CAPTIONER == 'llava':  get_llava()
elif CAPTIONER == 'blip2': get_blip2()
print('Setup complete.')


# ── CELL 2 — Load R-Bench Data + Images ─────────────────────
rbench_data = load_rbench()
pil_images, rbench_questions = load_images(rbench_data, n_images=N_IMAGES, seed=RANDOM_SEED)


# ── CELL 3 — Caption Generation (incremental) ───────────────
CAPTIONS_PATH = f'{SAVE_DIR}/{CAPTIONER}_captions.json'

if os.path.exists(CAPTIONS_PATH):
    with open(CAPTIONS_PATH) as f: captions = json.load(f)
    print(f'Loaded {len(captions)} cached captions')
else:
    captions = {}

new = 0
for img_id, pil in pil_images.items():
    if img_id not in captions:
        cap = caption_image(pil, captioner=CAPTIONER)
        if cap:
            captions[img_id] = cap
            new += 1
            print(f'  [{img_id}] {cap[:80]}')

if new > 0:
    with open(CAPTIONS_PATH, 'w') as f: json.dump(captions, f, indent=2)
print(f'{new} new, {len(captions)} total captions')


# ── CELL 4 — Inject Hallucinations from R-Bench GT=no ───────
INJECT_PATH = f'{SAVE_DIR}/injected_{CAPTIONER}.json'

if Path(INJECT_PATH).exists():
    with open(INJECT_PATH) as f: injected_data = json.load(f)
    print(f'Loaded {len(injected_data)} cached injections')
else:
    injected_data = {}
    for img_id, cap in captions.items():
        no_qs = [qa for qa in rbench_questions.get(img_id, [])
                 if qa['answer'].lower().strip() == 'no']
        if not no_qs: continue
        question = sorted(no_qs, key=lambda qa: len(qa['question']), reverse=True)[0]['question']
        statement = question_to_statement(question)
        if not statement: continue
        sep = ' ' if cap.rstrip().endswith('.') else '. '
        subj, rel, obj_ = parse_question(question)
        injected_data[img_id] = {
            'original_caption': cap, 'corrupted_caption': cap.rstrip() + sep + statement,
            'injected_question': question, 'injected_statement': statement,
            'subject': subj, 'relation': rel, 'object': obj_,
            'rel_type': classify_rel_type(question), 'gt': 'no',
        }
        print(f'  [{img_id}] {question}')
    with open(INJECT_PATH, 'w') as f: json.dump(injected_data, f, indent=2)

print(f'Injected: {len(injected_data)}/{len(captions)} images')


# ── CELL 5 — RelCheck Detection on Corrupted Captions ───────
from collections import defaultdict
print(f'Detecting on {len(injected_data)} corrupted triples...\n')

detection_results = []
counts = {'DETECTED': 0, 'MISSED': 0, 'UNCERTAIN': 0}

for img_id, inj in injected_data.items():
    pil = pil_images.get(img_id)
    if pil is None: continue
    s, r, o = inj['subject'], inj['relation'], inj['object']
    dets = detect_objects(pil, list(set([s, o])))
    verdict = verify_triple(s, r, o, dets, pil)
    status = {False: 'DETECTED', True: 'MISSED'}.get(verdict, 'UNCERTAIN')
    counts[status] += 1
    detection_results.append({'img_id': img_id, 'subject': s, 'relation': r,
        'object': o, 'rel_type': classify_relation(r), 'verdict': str(verdict), 'status': status})
    icon = {'DETECTED': '✓', 'MISSED': '✗', 'UNCERTAIN': '?'}[status]
    print(f'  [{img_id}] {icon} {status}: ({s}, {r}, {o})')

total = sum(counts.values())
print(f'\nRecall: {counts["DETECTED"]}/{total} ({100*counts["DETECTED"]/max(total,1):.1f}%)')
json.dump(detection_results, open(f'{SAVE_DIR}/detection_{CAPTIONER}.json', 'w'), indent=2)


# ── CELL 6 — Visual Knowledge Base Construction ─────────────
# Priority: cached run_600 KB → per-captioner KB → build fresh
KB_PATH = f'{SAVE_DIR}/kb_{CAPTIONER}.json'
KB_CACHED_PATH = '/content/drive/MyDrive/RelCheck_Data/run_600/knowledge_bases.json'

if os.path.exists(KB_PATH):
    with open(KB_PATH) as f: knowledge_bases = json.load(f)
    print(f'Loaded KB: {len(knowledge_bases)} images (from {KB_PATH})')
elif os.path.exists(KB_CACHED_PATH):
    with open(KB_CACHED_PATH) as f: _all_kb = json.load(f)
    knowledge_bases = {k: v for k, v in _all_kb.items() if k in pil_images}
    print(f'Loaded KB: {len(knowledge_bases)} images from run_600 cache ({len(_all_kb)} total)')
    del _all_kb
else:
    knowledge_bases = {}
    for idx, (img_id, img) in enumerate(pil_images.items()):
        t0 = time.time()
        kb = build_visual_kb(img, captions.get(img_id, ''), max_detections=20)
        knowledge_bases[img_id] = kb.to_dict() if hasattr(kb, 'to_dict') else kb
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f'  [{idx+1}/{len(pil_images)}] {img_id} ({time.time()-t0:.1f}s)')
        time.sleep(0.3)
    with open(KB_PATH, 'w') as f: json.dump(knowledge_bases, f)
    print(f'Saved KB for {len(knowledge_bases)} images')


# ── CELL 7 — Full RelCheck Correction Pipeline ──────────────
from relcheck_v2.correction._metrics import MetricsCollector

CORRECTED_PATH = f'{SAVE_DIR}/corrected_{CAPTIONER}.json'

mc = MetricsCollector()

if os.path.exists(CORRECTED_PATH):
    with open(CORRECTED_PATH) as f: correction_data = json.load(f)
    corrected_captions = {k: v['corrected'] for k, v in correction_data.items()}
    print(f'Loaded {len(corrected_captions)} cached corrections')
else:
    correction_data = {}
    corrected_captions = {}
    for idx, (img_id, inj) in enumerate(injected_data.items()):
        pil = pil_images.get(img_id)
        if pil is None: continue
        t0 = time.time()
        result = correct_long_caption(
            img_id, inj['corrupted_caption'], knowledge_bases.get(img_id, {}),
            pil_image=pil, cross_captions=None, metrics=mc)
        corrected = result.corrected
        corrected_captions[img_id] = corrected
        correction_data[img_id] = {
            'original': inj['original_caption'], 'corrupted': inj['corrupted_caption'],
            'corrected': corrected, 'errors': [e.triple.claim for e in result.errors],
            'edit_rate': result.edit_rate, 'status': result.status,
        }
        changed = corrected != inj['corrupted_caption']
        print(f'  [{idx+1}/{len(injected_data)}] [{img_id}] {"CHANGED" if changed else "same"} ({time.time()-t0:.1f}s)')
        if (idx + 1) % 10 == 0:
            with open(CORRECTED_PATH, 'w') as f: json.dump(correction_data, f, indent=2)
    with open(CORRECTED_PATH, 'w') as f: json.dump(correction_data, f, indent=2)
    n_changed = sum(1 for d in correction_data.values() if d['corrected'] != d['corrupted'])
    print(f'Done. {n_changed}/{len(correction_data)} modified.')

    # Save path logs and print summary
    mc.save(f'{SAVE_DIR}/path_logs.json')
    mc.print_summary()


# ── CELL 8 — R-POPE LLM-Judge Evaluation ────────────────────
results = run_synthetic_rpope(
    injected_data, corrected_captions, rbench_questions, verbose=True)
