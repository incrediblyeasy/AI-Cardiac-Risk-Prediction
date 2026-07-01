"""DS1/DS2 split integrity. The full leakage guard on extracted beats lands on
Day 2; this covers the patient-id lists themselves."""

from paper1_echofusenet.data import splits


def test_ds1_ds2_sizes():
    assert len(splits.DS1_PATIENTS) == 22
    assert len(splits.DS2_PATIENTS) == 22
    assert len(splits.EXCLUDED_PACED) == 4


def test_no_patient_overlap():
    assert set(splits.DS1_PATIENTS).isdisjoint(splits.DS2_PATIENTS)
    splits.assert_no_leakage()  # must not raise


def test_covers_all_48_records():
    combined = set(splits.DS1_PATIENTS) | set(splits.DS2_PATIENTS) | set(splits.EXCLUDED_PACED)
    assert len(combined) == 48
    assert combined == set(splits.ALL_MITDB_RECORDS)


def test_excluded_are_the_paced_records():
    assert set(splits.EXCLUDED_PACED) == {102, 104, 107, 217}


def test_split_for():
    assert splits.split_for(101) == "DS1"
    assert splits.split_for(100) == "DS2"
    assert splits.split_for(102) == "excluded"
