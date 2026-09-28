from hypothesis import settings


def test_registered_hypothesis_profiles_have_expected_limits() -> None:
    ci = settings.get_profile("ci")
    dev = settings.get_profile("dev")
    nightly = settings.get_profile("nightly")

    assert (
        ci.max_examples,
        ci.derandomize,
        ci.database,
        ci.deadline,
        ci.print_blob,
    ) == (
        100,
        True,
        None,
        None,
        True,
    )
    assert (dev.max_examples, dev.deadline) == (50, None)
    assert (nightly.max_examples, nightly.derandomize, nightly.deadline) == (
        2000,
        False,
        None,
    )
