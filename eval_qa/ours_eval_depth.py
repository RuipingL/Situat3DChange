import re
import numpy as np
UNIT_TO_METERS = {
    'm': 1.0,
    'meter': 1.0,
    'meters': 1.0,

    'cm': 0.01,
    'centimeter': 0.01,
    'centimeters': 0.01,

    'mm': 0.001,
    'millimeter': 0.001,
    'millimeters': 0.001,

    # You can add more units if needed:
    'in': 0.0254,
    'inch': 0.0254,
    'inches': 0.0254,

    'ft': 0.3048,
    'foot': 0.3048,
    'feet': 0.3048,
}

unit_pattern = "|".join(sorted(UNIT_TO_METERS.keys(), key=len, reverse=True))

distance_pattern = re.compile(
    rf'(\d+(?:\.\d+)?)[ ]*({unit_pattern})\b',  # number + recognized unit
    re.IGNORECASE
)

bare_number_pattern = re.compile(r'(\d+(?:\.\d+)?)\b')

NO_MOVEMENT_PHRASES = [
    "not moved",
    "no movement",
    "did not move",
    "unchanged",
    "still in place",
    "remain in place",
    "didn't budge",
    "unchanged position",
    "didn't relocate"
]


UNKNOWN_PHRASES = [
    "cannot determine",
    "not visible",
    "absent",
    "cannot measure",
    "not enough information",
    "unknown distance",
    "no data",
    "not enough visual information",
    "lack of reference",
    "lack of scale",
    "blurry",
    "distorted"
]

FUZZY_DISTANCE_PHRASES = {
    # "Within arm's reach" synonyms --> 0.5 meter horizontal distance
    "within arm reach": 0.5,
    "within arm's reach": 0.5,
    "within arms reach": 0.5,

    # General fuzzy terms --> approximate distances (in meters)
    "near": 1.0,
    "close": 1.0,
    "short distance": 1.0,
    "a few feet away": 1.0,
    "far": 3.0,
    "long distance": 5.0
}


def parse_distance_to_meters(sentence: str):
    """
    Parse a sentence for distance + unit (m, cm, mm, etc.) and return the distance in meters.
      - If a 'no movement' phrase appears, return 0.0 (indicating zero).
      - If there's a numeric + recognized unit, convert to meters and return that value.
      - If only a numeric value (no recognized unit), assume meters.
      - If a 'fuzzy distance' phrase appears, return the corresponding approximate distance in meters.
      - If an 'unknown' phrase appears, return None.
      - If nothing is found, return None.
    """
    lower_sentence = sentence.lower()

    # 1) Check for "no movement" phrases.
    if any(phrase in lower_sentence for phrase in NO_MOVEMENT_PHRASES):
        return 0.0

    # 2) Look for a numeric distance + recognized unit.
    match_with_unit = distance_pattern.search(sentence)
    if match_with_unit:
        numeric_str = match_with_unit.group(1)
        unit_str = match_with_unit.group(2).lower()
        value = float(numeric_str)
        multiplier = UNIT_TO_METERS.get(unit_str.rstrip('s'), 1.0)  
        return value * multiplier

    # 3) Check for "unknown" phrases.
    if any(phrase in lower_sentence for phrase in UNKNOWN_PHRASES):
        return None

    # 4) Check for "fuzzy distance" phrases.
    for phrase, distance in FUZZY_DISTANCE_PHRASES.items():
        if phrase in lower_sentence:
            return distance

    # 5) If no recognized unit was found but there's a bare number, assume meters.
    match_bare_number = bare_number_pattern.search(sentence)
    if match_bare_number:
        numeric_str = match_bare_number.group(1)
        value = float(numeric_str)
        return value  # Interpreted as meters

    # 6) Fallback if no recognized distance or special phrase is found.
    return None

def eval_depth_acc(pred_dict, gt_dict):
    acc = 0
    num = 0
    pred_dict = {int(k): parse_distance_to_meters(v[0]) for k, v in pred_dict.items()}
    gt_dict = {int(k): parse_distance_to_meters(v[0]) for k, v in gt_dict.items()}
    for index in pred_dict.keys():
        if pred_dict[index] is not None and gt_dict[index] is not None:
            if pred_dict[index] == 0 and gt_dict[index] == 0:
                acc += 1
                num += 1
            elif pred_dict[index] == 0 or gt_dict[index] == 0:
                acc += 0
                num += 1
            else:
                acc += 1- min((np.abs( pred_dict[index] - gt_dict[index])/gt_dict[index]), 1) #gt_dict[index]
                num += 1
    print('Accuracy:', acc/num)
    return acc/num
