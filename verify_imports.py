"""
verify_imports.py — Chapter 0 import verification script.
Run from inside the activated venv.
"""

libs = [
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("sklearn",     "scikit-learn"),
    ("xgboost",     "xgboost"),
    ("imblearn",    "imbalanced-learn"),
    ("shap",        "shap"),
    ("lime",        "lime"),
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("cv2",         "opencv-python"),
    ("easyocr",     "easyocr"),
    ("transformers","transformers"),
]

failed = []
for module_name, pkg_name in libs:
    try:
        __import__(module_name)
        print(f"  [OK]  {module_name} ({pkg_name})")
    except ImportError as e:
        print(f"  [FAIL] {module_name} ({pkg_name}): {e}")
        failed.append(module_name)

print()
if failed:
    print(f"FAILED imports: {failed}")
else:
    print("All imports verified successfully.")
