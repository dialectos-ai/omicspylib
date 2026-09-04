import textwrap
from collections.abc import Callable

# 1. Base parameters
COMMON_PARAMS: dict[str, str] = {
    "dataset": "dataset : TabularDataset\n    Dataset containing quantitative values.",
    "proteins_dataset": "dataset : ProteinsDataset\n    A proteins dataset object containing quantitative values.",
    "ax": "ax : Axes | None, default=None\n    Matplotlib Axes object. If ``None``, a new figure and axes are created.",
    "xlabel": "xlabel : str | None, default=None\n    Label for the x-axis.",
    "ylabel": "ylabel : str | None, default=None\n    Label for the y-axis.",
    "title": "title : str | None, default=None\n    Title of the plot.",
    "min_threshold": "min_threshold : float, default=0\n    Values below this threshold will be considered missing.",
    "returns_ax": "Returns\n-------\nAxes\n    A matplotlib Axes object containing the plot.",
}

# 2. Individual kwargs definitions (each indented as a list item)
COMMON_KWARGS: dict[str, str] = {
    "text_annotation_size": """\
* text_annotation_size : float, default=None
    Font size for text annotations.""",
    "text_annotation_rotation": """\
* text_annotation_rotation : float, default=0
    Rotation angle in degrees for text annotations.""",
    "text_annotation_round_digits": """\
* text_annotation_round_digits : int | None, default=None
    Number of decimal places to round text annotations to.""",
    "text_xlabel_rotation": """\
* text_xlabel_rotation : float | None, default=None
    Rotation angle in degrees for x-axis tick labels.""",
    "text_xlabel_ha": """\
* text_xlabel_ha : str | None, default=None
    Horizontal alignment for x-axis tick labels (e.g. 'left', 'center', 'right').""",
    "text_xlabel_va": """\
* text_xlabel_va : str | None, default=None
    Vertical alignment for x-axis tick labels (e.g. 'top', 'bottom', 'center').""",
    "show_experiment_names": """\
* show_experiment_names : bool, default=False
    If True, displays experiment names next to jitter points.""",
}


def build_kwargs_doc(*kwarg_keys: str) -> str:
    """Build a NumPy-style **kwargs documentation block for specific kwargs."""
    if not kwarg_keys:
        return ""

    items = []
    for key in kwarg_keys:
        if key in COMMON_KWARGS:
            items.append(COMMON_KWARGS[key])
        else:
            raise KeyError(f"Unknown kwarg doc key: '{key}'")

    body = "\n\n".join(items)
    # Indent bullet points under the **kwargs parameter
    indented_body = textwrap.indent(body, "    ")

    return f"**kwargs : dict, optional\n    Additional keyword arguments:\n\n{indented_body}"


def doc(*args_to_include_kwargs: str, **custom_replacements: str) -> Callable:
    """
    Decorator to inject parameter and kwarg documentation.

    Usage:
        @doc("text_annotation_size", "text_xlabel_rotation")  # Select specific kwargs
        def my_plot(...):
            '''
            {dataset}
            {kwargs_doc}
            {returns_ax}
            '''
    """

    def decorator(func: Callable) -> Callable:
        if func.__doc__:
            replacements = {**COMMON_PARAMS, **custom_replacements}

            # If positional string arguments are passed, build {kwargs_doc} from them
            if args_to_include_kwargs:
                replacements["kwargs_doc"] = build_kwargs_doc(*args_to_include_kwargs)

            func.__doc__ = func.__doc__.format_map(replacements)
        return func

    return decorator
