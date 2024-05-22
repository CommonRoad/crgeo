from pathlib import Path

import pytest

from commonroad_geometric.common.io_extensions.scenario_files import filter_max_scenarios, filter_scenario_filenames, filter_scenario_paths, filter_scenario_subvariants


@pytest.fixture()
def base_path() -> Path:
    return Path('does', 'not', 'exist')


@pytest.fixture()
def excluded_path_1(base_path) -> Path:
    return base_path / base_path / 'DEU_ScenarioA_1.xml'


@pytest.fixture()
def excluded_path_2(base_path) -> Path:
    return base_path / 'FRA_ScenarioC_2_T-1.xml'


@pytest.fixture()
def xml_paths(base_path, excluded_path_1, excluded_path_2) -> list[Path]:
    xml_paths = [
        excluded_path_1,
        base_path / 'DEU_ScenarioA_2.xml',
        base_path / 'ESP_ScenarioB_1.xml',
        base_path / 'ESP_ScenarioB_3.xml',
        base_path / 'FRA_ScenarioC_1_T-1.xml',
        excluded_path_2
    ]
    return xml_paths


@pytest.fixture()
def protobuf_paths(xml_paths) -> list[Path]:
    protobuf_paths = []
    for xml_path in xml_paths:
        protobuf_path = xml_path.with_suffix('.proto')
        protobuf_paths.append(protobuf_path)
    return protobuf_paths


def test_filter_scenario_paths(xml_paths, excluded_path_1):
    filtered_scenario_paths = filter_scenario_paths(
        scenario_paths=xml_paths,
        excluded_scenario_names={excluded_path_1.stem},  # Excludes first path in xml_paths
        subvariant_prefix_regex=r'_\d',  # Matches digits, excludes base_path / 'ESP_ScenarioB_3.xml' and excluded_path_2
        max_scenarios=2  # Excludes base_path / 'FRA_ScenarioC_1_T-1.xml'
    )
    assert len(filtered_scenario_paths) == 2
    assert filtered_scenario_paths == xml_paths[1:3]


@pytest.mark.parametrize("max_scenarios", [x for x in range(0, 10)])
def test_filter_max_scenarios(xml_paths, max_scenarios):
    filtered_scenario_paths = filter_max_scenarios(scenario_paths=xml_paths, max_scenarios=max_scenarios)

    assert len(filtered_scenario_paths) == min(6, max_scenarios)  # There are 6 scenario_paths in paths_xml
    assert filtered_scenario_paths == xml_paths[:max_scenarios]


def test_filter_scenario_filenames_xml(xml_paths, excluded_path_1, excluded_path_2):
    excluded_scenario_names = {excluded_path_1.stem, excluded_path_2.stem}
    filtered_scenario_paths = filter_scenario_filenames(scenario_paths=xml_paths, excluded_scenario_names=excluded_scenario_names)

    assert len(filtered_scenario_paths) == len(xml_paths) - len(excluded_scenario_names)
    assert excluded_path_1 not in filtered_scenario_paths
    assert excluded_path_2 not in filtered_scenario_paths


def test_filter_scenario_filenames_ignores_file_extension(
    xml_paths,
    protobuf_paths,
    excluded_path_1,
    excluded_path_2
):
    excluded_scenario_names = {excluded_path_1.stem, excluded_path_2.stem}
    filtered_scenario_paths_xml = filter_scenario_filenames(scenario_paths=xml_paths, excluded_scenario_names=excluded_scenario_names)
    filtered_scenario_paths_protobuf = filter_scenario_filenames(scenario_paths=protobuf_paths, excluded_scenario_names=excluded_scenario_names)

    assert len(filtered_scenario_paths_protobuf) == len(protobuf_paths) - len(excluded_scenario_names)
    assert len(filtered_scenario_paths_xml) == len(filtered_scenario_paths_protobuf)
    assert excluded_path_1 not in filtered_scenario_paths_xml
    assert excluded_path_2 not in filtered_scenario_paths_xml
    assert excluded_path_1.with_suffix('.proto') not in filtered_scenario_paths_protobuf
    assert excluded_path_2.with_suffix('.proto') not in filtered_scenario_paths_protobuf


def test_filter_scenario_subvariants(xml_paths):
    filtered_scenario_paths = filter_scenario_subvariants(scenario_paths=xml_paths)
    # There's two variants of every scenario
    # Structure of xml_paths: 3 * 2 variants, get every other element

    assert len(filtered_scenario_paths) == len(xml_paths) // 2

    for first_variant in xml_paths[::2]:
        assert first_variant in filtered_scenario_paths
    for second_variant in xml_paths[1::2]:
        assert second_variant not in filtered_scenario_paths


def test_filter_scenario_subvariants_ignores_file_extension(xml_paths, protobuf_paths):
    filtered_scenario_paths_xml = filter_scenario_subvariants(scenario_paths=xml_paths)
    filtered_scenario_paths_protobuf = filter_scenario_subvariants(scenario_paths=protobuf_paths)
    # There's two variants of every scenario
    # Structure of xml_paths: 3 * 2 variants, get every other element

    assert len(filtered_scenario_paths_protobuf) == len(protobuf_paths) // 2
    assert len(filtered_scenario_paths_protobuf) == len(filtered_scenario_paths_xml)

    for xml_variant, protobuf_variant in zip(filtered_scenario_paths_xml, filtered_scenario_paths_protobuf):
        assert protobuf_variant == xml_variant.with_suffix('.proto')

    for first_variant in protobuf_paths[::2]:
        assert first_variant in filtered_scenario_paths_protobuf
    for second_variant in protobuf_paths[1::2]:
        assert second_variant not in filtered_scenario_paths_protobuf
