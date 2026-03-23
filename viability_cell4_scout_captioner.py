# ── Cell 4: Generate captions via Llama-4-Scout (richer than BLIP-2) ─────
import base64, io

CAPTION_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

def image_to_base64(img):
    """Convert PIL image to base64 string for API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

captions = {}
for img_id, img in images.items():
    b64 = image_to_base64(img)
    resp = together_client.chat.completions.create(
        model=CAPTION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": "Describe this image in one detailed paragraph. "
                         "Focus on the spatial relationships between objects "
                         "(what is on, next to, behind, in front of what), "
                         "the actions people or animals are performing, "
                         "and any notable attributes. Be specific and factual."}
            ]
        }],
        max_tokens=256,
        temperature=0.3,
    )
    caption = resp.choices[0].message.content.strip()
    captions[img_id] = caption
    print(f"[{img_id}] {caption[:120]}...")

print(f"\n✓ {len(captions)} captions generated via {CAPTION_MODEL}")
