import multiscale_run.utils as mu


def test_timestamps():
    assert list(mu.timesteps(10, 1)) == list(range(1, 11, 1))
    assert list(mu.timesteps(10, 0.9)) == [i * 0.9 for i in range(1, 12, 1)]


def test_check_param():
    assert mu.check_param([], 0) == ""
    assert mu.check_param([0], 0) == ""
    assert mu.check_param([3, 0.1, -1, 0, 0.0], 0) == ""
    assert mu.check_param([float("NaN")], 0) != ""
    assert mu.check_param([float("inf")], 0) != ""
    assert mu.check_param([float("NaN"), 1], 0) != ""
    assert mu.check_param([float("inf"), 0], 0) != ""
