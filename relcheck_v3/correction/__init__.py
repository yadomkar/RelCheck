"""Correction stage — hallucination identification and surgical caption editing."""


def __getattr__(name: str):
    if name == "RelCheckCaptionEditor":
        from relcheck_v3.correction.editor import RelCheckCaptionEditor
        return RelCheckCaptionEditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RelCheckCaptionEditor"]
