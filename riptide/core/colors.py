from enum import Enum

COLORS = {
    "BKG": {
        "name": "magenta",
        "hex": "#FF00FF",
        "rgb": (255, 0, 255),
    },
    "CLS": {
        "name": "crimson",
        "hex": "#DC143C",
        "rgb": (220, 20, 60),
    },
    "LOC": {
        "name": "gold",
        "hex": "#FFD700",
        "rgb": (255, 215, 0),
    },
    "CLL": {
        "name": "darkorange",
        "hex": "#FF8C00",
        "rgb": (255, 215, 0),
    },
    "DUP": {
        "name": "cyan",
        "hex": "#00FFFF",
        "rgb": (0, 255, 255),
    },
    "MIS": {
        "name": "yellowgreen",
        "hex": "#9ACD32",
        "rgb": (154, 205, 50),
    },
    "TP": {
        "name": "lime",
        "hex": "#00FF00",
        "rgb": (0, 255, 0),
    },
    "BST": {
        "name": "blue",
        "hex": "#0000FF",
        "rgb": (0, 0, 255),
    },
    "FN": {
        "name": "olivedrab",
        "hex": "#6B8E23",
        "rgb": (107, 142, 35),
    },
    "FP": {
        "name": "red",
        "hex": "#FF0000",
        "rgb": (255, 0, 0),
    },
    "default": {
        "name": "white",
        "hex": "#FFFFFF",
        "rgb": (255, 255, 255),
    },
}


class ErrorColor(str, Enum):

    BKG = "BKG"
    CLS = "CLS"
    LOC = "LOC"
    CLL = "CLL"
    DUP = "DUP"
    MIS = "MIS"
    TP = "TP"
    FN = "FN"
    BST = "BST"
    FP = "FP"
    WHITE = "default"

    @classmethod
    def _missing_(cls, value):
        return cls.WHITE

    @property
    def colorstr(self):
        return COLORS[self.value]["name"]

    @property
    def hex(self):
        return COLORS[self.value]["hex"]

    def rgb(self, alpha: int = 0, as_tuple: bool = True):
        assert 0 <= alpha <= 255, "Alpha must be between 0 and 255"
        color = (
            COLORS[self.value]["rgb"] + (alpha,) if alpha else COLORS[self.value]["rgb"]
        )
        if not as_tuple:
            color = f"rgba{color}" if alpha else f"rgb{color}"
        return color
