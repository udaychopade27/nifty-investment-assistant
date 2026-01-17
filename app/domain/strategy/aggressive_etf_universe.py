AGGRESSIVE_ETFS = {
    "NIFTYBEES": {
        "category": "LARGE_CAP",
        "role": "Stability anchor",
        "weight": 0.20,
    },
    "JUNIORBEES": {
        "category": "NEXT_50",
        "role": "Growth accelerator",
        "weight": 0.30,
    },
    "MIDCAPETF": {
        "category": "MID_CAP",
        "role": "Core growth",
        "weight": 0.25,
    },
    "SMALLCAPETF": {
        "category": "SMALL_CAP",
        "role": "Alpha generator",
        "weight": 0.15,
    },
    "GOLDBEES": {
        "category": "GOLD",
        "role": "Crash hedge",
        "weight": 0.10,
    },
}


def get_target_weights() -> dict:
    return {k: v["weight"] for k, v in AGGRESSIVE_ETFS.items()}
