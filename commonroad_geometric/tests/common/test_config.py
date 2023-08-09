from commonroad_geometric.common.config import Config, ImmutableError


def test_config() -> None:
    d1 = {
        "a": {
            "test_int": 42,
            "test_str": "hello",
        },
        "b": None,
        "ref": "$a.test_int",
        "ref2": "$a.test_str",
        "ref3": "$c",
    }
    cfg1 = Config(d1.copy())
    cfg2 = Config({
        "a": {
            "test_int": 420,
            "new": "value",
        },
        "c": "test",
    })

    assert isinstance(cfg1.a, Config)
    assert cfg1.a is cfg1.get("a") and cfg1.a is cfg1["a"]
    assert cfg1.a.test_int == 42
    assert cfg1.value_by_path(("a", "test_str")) == "hello"
    assert cfg1["b"] is None
    assert cfg1.as_dict() == d1
    assert cfg1.get("nope", default="default value") == "default value"
    assert cfg1.ref == 42
    assert cfg1.ref2 == "hello"
    try:
        cfg1.ref3
        assert False
    except KeyError:
        pass

    cfg_overlay = cfg1.overlay(cfg2)
    assert isinstance(cfg_overlay.a, Config)
    assert cfg_overlay.a.test_int == 420
    assert cfg_overlay.a.test_str == "hello"
    assert cfg_overlay.a.new == "value"
    assert cfg_overlay.b is None
    assert cfg_overlay.c == "test"
    assert cfg_overlay.ref == 420
    assert cfg_overlay.ref == 420
    assert cfg_overlay.ref2 == "hello"
    assert cfg_overlay.ref3 == "test"

    try:
        cfg1.b = 1
        assert False
    except ImmutableError:
        pass
    try:
        cfg1["b"] = 1
        assert False
    except ImmutableError:
        pass
    try:
        x = cfg1.nope
        assert False
    except KeyError:
        pass

    cfg1_mut = cfg1.mutable()
    cfg1_mut.a.test_int = 70
    assert cfg1_mut.ref == 70
