from sentinel.windowing import n_windows


def test_n_windows_basic():
    assert n_windows(5.0, 1.0, 0.5) == 9
    assert n_windows(1.0, 1.0, 0.5) == 1
    assert n_windows(1.49, 1.0, 0.5) == 1
    assert n_windows(1.5, 1.0, 0.5) == 2


def test_n_windows_too_short():
    assert n_windows(0.99, 1.0, 0.5) == 0
    assert n_windows(0.0) == 0


def test_n_windows_long_recording():
    assert n_windows(3600.0, 1.0, 0.5) == 7199
