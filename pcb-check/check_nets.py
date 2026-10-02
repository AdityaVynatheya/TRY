"""Extract the netlist from the isolation-routed copper mask and check it
against the intended ESP32-S3-CAM + INMP441 + MAX98357A schematic.

In copper_mask.png white is copper and black is etched away, so every
connected white region is one net. Usage: python3 check_nets.py
Requires: pillow, numpy, scipy.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

IMG = Path(__file__).with_name("copper_mask.png")

im = np.array(Image.open(IMG).convert("L"))
cu = im > 128
cu[:12, :] = cu[-12:, :] = cu[:, :12] = cu[:, -12:] = False  # board outline
# Drill holes are small black islands inside pads: fill them so each pad is one blob.
holes, nh = ndimage.label(~cu)
hole_sizes = ndimage.sum(~cu, holes, range(1, nh + 1))
cu |= np.isin(holes, 1 + np.where(hole_sizes < 200)[0])
lab, _ = ndimage.label(cu)


def net_at(x, y):
    for r in range(12):
        win = lab[y - r:y + r + 1, x - r:x + r + 1]
        v = win[win > 0]
        if v.size:
            return int(np.bincount(v).argmax())
    return 0


# Pad centres in image pixels (copper-side view).
LEFT = ["TX", "RX", "1", "2", "42", "41", "40", "39", "38", "37",
        "36", "35", "0", "45", "48", "47", "21", "20", "19", "GND"]
RIGHT = ["3V3", "EN", "4", "5", "6", "7", "15", "16", "17", "18",
         "8", "3", "46", "9", "10", "11", "12", "13", "14", "5V"]
PADS = {}
for i, (a, b) in enumerate(zip(LEFT, RIGHT)):
    y = int(round(36 + i * 30.8))
    PADS["ESP." + a] = (512, y)
    PADS["ESP." + b] = (853, y)
PADS.update({
    "MIC.SD": (318, 128), "MIC.SCK": (408, 128),
    "MIC.VDD": (318, 158), "MIC.WS": (408, 158),
    "MIC.GND": (318, 188), "MIC.LR": (408, 188),
    "AMP.LRC": (255, 465), "AMP.BCLK": (255, 496), "AMP.DIN": (255, 526),
    "AMP.GAIN": (255, 557), "AMP.SD": (255, 588), "AMP.GND": (255, 619),
    "AMP.VIN": (255, 650),
    "JP.a": (312, 588), "JP.b": (343, 588), "JP.c": (374, 588), "JP.d": (405, 588),
    "R1.2": (312, 619),
    "C1.+": (255, 706), "C1.-": (208, 706), "C2.a": (255, 754), "C2.b": (208, 754),
    "C470.+": (463, 802), "C470.-": (463, 742),
    "SPK-.a": (60, 712), "SPK-.b": (130, 712),
    "SPK+.a": (60, 770), "SPK+.b": (130, 770),
    "SW.1": (918, 685), "SW.2": (998, 685), "SW.3": (918, 740), "SW.4": (998, 740),
})

# Intended connectivity. Every pad not listed here must be on its own net.
EXPECTED = {
    "GND": {"ESP.GND", "MIC.GND", "MIC.LR", "AMP.GND", "JP.d", "R1.2",
            "C1.-", "C2.b", "C470.-", "SW.4"},
    "5V": {"ESP.5V", "AMP.VIN", "C1.+", "C2.a", "C470.+"},
    "3V3": {"ESP.3V3", "MIC.VDD"},
    "MIC_SD": {"ESP.1", "MIC.SD"},
    "MIC_SCK": {"ESP.2", "MIC.SCK"},
    "MIC_WS": {"ESP.42", "MIC.WS"},
    "AMP_LRC": {"ESP.41", "AMP.LRC"},
    "AMP_BCLK": {"ESP.47", "AMP.BCLK"},
    "AMP_DIN": {"ESP.21", "AMP.DIN"},
    "TALK": {"ESP.14", "SW.1"},
    "GAIN": {"AMP.GAIN", "JP.c"},
    "GAIN_15DB": {"JP.a", "JP.b"},
    "SPK-": {"SPK-.a", "SPK-.b"},
    "SPK+": {"SPK+.a", "SPK+.b"},
}

found = {}
for name, (x, y) in PADS.items():
    found.setdefault(net_at(x, y), set()).add(name)
actual = {frozenset(v) for v in found.values()}

listed = set().union(*EXPECTED.values())
want = {frozenset(v) for v in EXPECTED.values()}
want |= {frozenset([p]) for p in PADS if p not in listed}

for net in sorted(actual, key=lambda s: (-len(s), sorted(s))):
    if len(net) > 1:
        print(" + ".join(sorted(net)))

missing, extra = want - actual, actual - want
for net in missing:
    print("MISSING / BROKEN:", sorted(net))
for net in extra:
    print("UNEXPECTED NET:", sorted(net))
if missing or extra:
    sys.exit(1)
print("OK: copper matches the intended netlist, no shorts between ESP pins.")
