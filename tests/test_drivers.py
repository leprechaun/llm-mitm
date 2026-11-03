from llm_mitm.drivers import Usage


def test_usage_update_copies_all_fields_in_place():
    usage = Usage(prompt_tokens=1, completion_tokens=2, cache_read_tokens=3, cache_write_tokens=4, service_tier="x")
    other = Usage(prompt_tokens=10, completion_tokens=20, cache_read_tokens=30, cache_write_tokens=40, service_tier="flex")

    usage.update(other)

    assert usage == other
    # Updates the same object in place -- callers holding a reference to
    # the original `usage` see the change without reassignment.
    assert usage is not other
