# ============================================================
# Cell 3b -- Multi-Model Captioning (LLaVA-1.5 + Qwen3-VL-8B)
# ============================================================
# Generates detailed captions from multiple MLLMs for the same images.
# Each model is loaded, used, then unloaded to fit T4 GPU memory.
# Uses "describe in detail" prompt to elicit rich relational descriptions.
# Checkpoint: loads from Drive if already computed.
#
# IMPORTANT: Run this AFTER Cell 3 (BLIP-2 captioning) and BEFORE Cell 4 (KB).
# BLIP-2 must be unloaded first to free GPU memory.

import gc
import base64
from io import BytesIO

def encode_b64(image):
    """Encode PIL image to base64 JPEG string for API calls."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

LLAVA_CAPTIONS_PATH = f"{SAVE_DIR}/llava_captions.json"

DESCRIBE_PROMPT = "Describe this image in detail. Include all objects, their spatial positions relative to each other, any actions or interactions taking place, and notable attributes like colors and sizes."

# ── Unload BLIP-2 to free GPU ──
try:
    del blip2_model, blip2_processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Unloaded BLIP-2 to free GPU memory.")
except:
    print("BLIP-2 not loaded, skipping unload.")

# ============================
# LLaVA-1.5-7B (4-bit quantized)
# ============================
llava_captions = load_checkpoint(LLAVA_CAPTIONS_PATH) or {}

if len(llava_captions) >= len(images):
    print(f"LLaVA captions: loaded {len(llava_captions)} from cache.")
else:
    print("Loading LLaVA-1.5-7B (4-bit)...")
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print("LLaVA loaded.")

    todo = [img_id for img_id in images if img_id not in llava_captions]
    print(f"Captioning {len(todo)} images with LLaVA...")

    for idx, img_id in enumerate(todo):
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": DESCRIBE_PROMPT},
            ]},
        ]
        prompt_text = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = llava_processor(images=images[img_id], text=prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = llava_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        decoded = llava_processor.decode(output[0], skip_special_tokens=True)

        # Extract only the assistant response
        if "ASSISTANT:" in decoded:
            caption = decoded.split("ASSISTANT:")[-1].strip()
        else:
            caption = decoded.strip()

        llava_captions[img_id] = caption

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(todo)}] {img_id}: {caption[:80]}...")
            save_checkpoint(llava_captions, LLAVA_CAPTIONS_PATH)

    save_checkpoint(llava_captions, LLAVA_CAPTIONS_PATH)
    print(f"LLaVA captioning done: {len(llava_captions)} images.")

    # Unload LLaVA
    del llava_model, llava_processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Unloaded LLaVA.")


# ============================
# Qwen3-VL-8B (via Together.ai — serverless)
# ============================
# Second VLM captioner: Qwen3-VL-8B (8B, weaker than Maverick 17B MoE).
# Generates detailed captions via Together.ai serverless API.
# Smaller model = more hallucination-prone = better for demonstrating correction.

SCOUT_CAPTIONS_PATH = f"{SAVE_DIR}/scout_captions.json"  # kept for checkpoint compat
SCOUT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # serverless on Together.ai

scout_captions = load_checkpoint(SCOUT_CAPTIONS_PATH) or {}

if len(scout_captions) >= len(images):
    print(f"Qwen3-VL-8B captions: loaded {len(scout_captions)} from cache.")
else:
    todo = [img_id for img_id in images if img_id not in scout_captions]
    print(f"Captioning {len(todo)} images with Qwen3-VL-8B (via Together.ai)...")

    for idx, img_id in enumerate(todo):
        b64 = encode_b64(images[img_id])
        raw = llm_call(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            model=SCOUT_MODEL,
            max_tokens=256,
        )
        scout_captions[img_id] = raw.strip() if raw else ""

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(todo)}] {img_id}: {raw[:80] if raw else 'EMPTY'}...")
            save_checkpoint(scout_captions, SCOUT_CAPTIONS_PATH)
        time.sleep(0.3)

    save_checkpoint(scout_captions, SCOUT_CAPTIONS_PATH)
    print(f"Qwen3-VL-8B captioning done: {len(scout_captions)} images.")


# ── Summary ──
print(f"\n=== Caption Summary ===")
for name, caps in [("BLIP-2", captions), ("LLaVA-1.5", llava_captions), ("Qwen3-VL-8B", scout_captions)]:
    lengths = [len(c) for c in caps.values() if c]
    if lengths:
        print(f"  {name}: {len(caps)} captions, avg {np.mean(lengths):.0f} chars, avg {np.mean([len(c.split('.')) for c in caps.values() if c]):.1f} sentences")
