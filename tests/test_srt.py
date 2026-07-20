from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SRT_PATH = Path(__file__).resolve().parents[1] / "omnivoice" / "utils" / "srt.py"
_SPEC = spec_from_file_location("omnivoice_utils_srt", _SRT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_srt_content = _MODULE.build_srt_content
split_text_into_srt_cues = _MODULE.split_text_into_srt_cues


def test_split_text_into_srt_cues_uses_sentence_boundaries():
    text = "Hello world. This is a test. Another one."

    assert split_text_into_srt_cues(text) == [
        "Hello world.",
        "This is a test.",
        "Another one.",
    ]


def test_split_text_into_srt_cues_splits_long_sentence_by_words():
    text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone "
        "twentytwo twentythree twentyfour twentyfive twentysix twentyseven "
        "twentyeight twentynine thirty"
    )

    cues = split_text_into_srt_cues(text)

    assert len(cues) == 3
    assert [len(cue.split()) for cue in cues] == [10, 10, 10]
    assert " ".join(cues).replace("  ", " ").strip() == text


def test_split_text_into_srt_cues_merges_short_trailing_fragment():
    text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone "
        "twentytwo twentythree"
    )

    cues = split_text_into_srt_cues(text)

    assert len(cues) == 2
    assert [len(cue.split()) for cue in cues] == [11, 12]
    assert " ".join(cues).replace("  ", " ").strip() == text


def test_build_srt_content_formats_subtitles():
    text = "Hello world. This is a test."
    srt = build_srt_content(text, duration_s=4.0)

    assert "1\n" in srt
    assert "2\n" in srt
    assert "Hello world." in srt
    assert "This is a test." in srt
