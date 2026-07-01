"""AAMI mapping correctness against the de Chazal (2004) grouping."""

from paper1_echofusenet.data import aami


def test_five_classes_in_canonical_order():
    assert aami.AAMI_CLASSES == ("N", "S", "V", "F", "Q")


def test_every_mapped_symbol_lands_in_a_valid_class():
    assert set(aami.SYMBOL_TO_AAMI.values()) <= set(aami.AAMI_CLASSES)


def test_representative_symbols():
    cases = {
        "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V",
        "F": "F",
        "/": "Q", "f": "Q", "Q": "Q",
    }
    for symbol, expected in cases.items():
        assert aami.symbol_to_aami(symbol) == expected, symbol


def test_non_beat_symbol_returns_none():
    assert aami.symbol_to_aami("+") is None
    assert aami.symbol_to_aami("~") is None


def test_class_index_roundtrip():
    for i, cls in enumerate(aami.AAMI_CLASSES):
        assert aami.class_index(cls) == i
