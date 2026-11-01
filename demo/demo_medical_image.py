"""
demo/demo_medical_image.py
Runs only if models/medical_image/resnet18_xray.pt exists.
Otherwise prints SKIPPED and exits 0.
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_SLIM = os.path.join(PROJECT_ROOT, "models", "medical_image", "resnet18_xray_weights.pt")
CKPT_ORIG = os.path.join(PROJECT_ROOT, "models", "medical_image", "resnet18_xray.pt")
CKPT = CKPT_SLIM if os.path.exists(CKPT_SLIM) else CKPT_ORIG

print("=" * 60)
print("  MediTwin AI — Medical Image Agent Demo")
print("=" * 60)

if not os.path.exists(CKPT):
    print(f"\n  not available: checkpoint missing ({CKPT})")
    print("  [demo_medical_image SKIPPED]")
    sys.exit(0)

sys.path.insert(0, PROJECT_ROOT)
import time, glob
from agents.medical_image_agent.medical_image_agent import predict_image

TEST_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia"
    "/versions/2/chest_xray/test")
imgs = []
for sub in ["NORMAL", "PNEUMONIA"]:
    imgs += sorted(glob.glob(os.path.join(TEST_DIR, sub, "*.jpeg")))[:1]

print(f"\n  Checkpoint: {CKPT}")
print(f"  Running inference on {len(imgs)} test images …\n")
for img in imgs:
    t0 = time.time()
    result = predict_image(img)
    elapsed = time.time() - t0
    print(f"  Image      : {os.path.basename(img)}")
    print(f"  Prediction : {result.get('prediction', result)}")
    print(f"  Time       : {elapsed:.2f}s\n")

print("  Demo complete. [demo_medical_image PASS]")
