from pathlib import Path

from commonroad_geometric.common.io_extensions.scenario_files import ScenarioFileFormat, find_scenario_paths


def test_find_scenario_path_xml(arg_carcarana_path_xml):
    scenario_files = find_scenario_paths(directory=arg_carcarana_path_xml, file_format=ScenarioFileFormat.XML)
    assert scenario_files == [arg_carcarana_path_xml]


def test_find_scenario_paths_xml(scenario_directory_xml, expected_paths_xml):
    scenario_files = find_scenario_paths(directory=scenario_directory_xml, file_format=ScenarioFileFormat.XML)
    assert set(scenario_files) == set(expected_paths_xml)


def test_find_scenario_paths_xml_sorted(scenario_directory_xml, expected_paths_xml):
    scenario_files = find_scenario_paths(directory=scenario_directory_xml, file_format=ScenarioFileFormat.XML)
    assert scenario_files == expected_paths_xml


def test_find_scenario_path_pkl(arg_carcarana_pkl):
    scenario_files = find_scenario_paths(directory=arg_carcarana_pkl, file_format=ScenarioFileFormat.BUNDLE)
    assert scenario_files == [arg_carcarana_pkl]


def test_find_scenario_paths_pkl(scenario_directory_pickle, expected_paths_pkl):
    scenario_files = find_scenario_paths(directory=scenario_directory_pickle, file_format=ScenarioFileFormat.BUNDLE)
    assert set(scenario_files) == set(expected_paths_pkl)


def test_find_scenario_paths_pkl_sorted(scenario_directory_pickle, expected_paths_pkl):
    scenario_files = find_scenario_paths(directory=scenario_directory_pickle, file_format=ScenarioFileFormat.BUNDLE)
    assert scenario_files == expected_paths_pkl


def test_find_scenario_paths_all_sorted(
    root_testdata_path,
    expected_paths_pkl
):
    # This test will break if we add more scenarios to the testdata
    scenario_files = find_scenario_paths(directory=root_testdata_path, file_format=ScenarioFileFormat.ALL)
    # Should only find the pkl files, as we prefer loading these
    assert scenario_files == expected_paths_pkl


def test_find_scenario_paths_xml_empty():
    xml_scenario_files = find_scenario_paths(directory=Path('nonsense.xml'), file_format=ScenarioFileFormat.XML)
    only_pkl_scenario_files = find_scenario_paths(directory=Path('nonsense.pkl'), file_format=ScenarioFileFormat.BUNDLE)
    all_pkl_scenario_files = find_scenario_paths(directory=Path('nonsense.pkl'), file_format=ScenarioFileFormat.ALL)

    assert xml_scenario_files == []
    assert only_pkl_scenario_files == []
    assert all_pkl_scenario_files == []
