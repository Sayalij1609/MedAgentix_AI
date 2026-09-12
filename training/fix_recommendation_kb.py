# -*- coding: utf-8 -*-
"""
Fix recommendation_knowledge.json:
1. Clean stringified Python lists into real lists
2. Deduplicate medications/diets/workouts
"""
import json, re, ast, os

PATH = "models/recommendation_model/data/recommendation_knowledge.json"

print(f"Loading {PATH}...")
with open(PATH, encoding='utf-8') as f:
    kb = json.load(f)

print(f"  Diseases: {len(kb)}")

def clean_list_field(field_val):
    """Convert any field value to a clean list of strings."""
    if not field_val:
        return []
    result = []
    for item in field_val:
        item_str = str(item).strip()
        if not item_str or item_str in ('nan', 'None', ''):
            continue
        # Check if item is a stringified Python list
        if item_str.startswith('[') and item_str.endswith(']'):
            try:
                parsed = ast.literal_eval(item_str)
                if isinstance(parsed, list):
                    result.extend([str(x).strip() for x in parsed if str(x).strip() not in ('', 'nan', 'None')])
                    continue
            except Exception:
                # Remove brackets and split by comma
                inner = item_str[1:-1]
                parts = [p.strip().strip("'\"") for p in inner.split("',") if p.strip()]
                result.extend([p for p in parts if p and p not in ('nan', 'None')])
                continue
        result.append(item_str)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in result:
        if x.lower() not in seen:
            seen.add(x.lower())
            deduped.append(x)
    return deduped

fixed = 0
for disease, info in kb.items():
    original = json.dumps(info)
    info['medications'] = clean_list_field(info.get('medications', []))
    info['diet_recommendations'] = clean_list_field(info.get('diet_recommendations', []))
    info['workout_recommendations'] = clean_list_field(info.get('workout_recommendations', []))
    info['precautions'] = clean_list_field(info.get('precautions', []))
    info['diagnostic_tests'] = clean_list_field(info.get('diagnostic_tests', []))
    if json.dumps(info) != original:
        fixed += 1

print(f"  Fixed {fixed} disease entries")

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

size = os.path.getsize(PATH) / 1024
print(f"  Saved: {PATH} ({size:.1f} KB)")

# Quick verify
test_diseases = ['gastroenteritis', 'pneumonia', 'diabetes', 'infectious gastroenteritis']
print("\n  Spot-check:")
for d in test_diseases:
    info = kb.get(d, {})
    meds = info.get('medications', [])[:2]
    print(f"  {d}: meds={meds}")

print("\n  Done - recommendation KB cleaned!")
