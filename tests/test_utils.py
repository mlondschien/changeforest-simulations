import pytest

from changeforest_simulations.utils import string_to_kwargs


@pytest.mark.parametrize(
    "string, kwargs",
    [
        (
            "method__minimal_relative_segment_length=0.01",
            ("method", {"minimal_relative_segment_length": 0.01}),
        ),
        ("value__n_observations=10", ("value", {"n_observations": 10})),
        ("method__some=value", ("method", {"some": "value"})),
        (
            "method__n_observations=10__some=value",
            ("method", {"some": "value", "n_observations": 10}),
        ),
    ],
)
def test_string_to_kwargs(string, kwargs):
    assert string_to_kwargs(string) == kwargs
