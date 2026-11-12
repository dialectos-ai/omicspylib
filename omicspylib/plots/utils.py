# utility plotting functions
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def apply_xtick_formatting(
        ax: Axes,
        rotation: float | None = None,
        ha: str | None = None,
        va: str | None = None,
        default_rotation: float = 0,
) -> Axes:
    """Apply rotation and alignment settings to x-axis tick labels."""
    rot = rotation if rotation is not None else default_rotation
    horizontal_align = ha if ha is not None else ("right" if rot != 0 else "center")

    kwargs = {"rotation": rot, "ha": horizontal_align}
    if va is not None:
        kwargs["va"] = va

    plt.setp(ax.get_xticklabels(), **kwargs)
    return ax

def apply_text_annotation_formatting(
    ax: Axes,
    rotation: float = 0,
    round_digits: int | None = None,
) -> Axes:
    """Apply rotation and numeric rounding to text annotations on axes."""
    for txt in ax.texts:
        if rotation != 0:
            txt.set_rotation(rotation)
        if round_digits is not None:
            try:
                val = float(txt.get_text())
                txt.set_text(f"{val:.{round_digits}f}")
            except ValueError:
                pass
    return ax
