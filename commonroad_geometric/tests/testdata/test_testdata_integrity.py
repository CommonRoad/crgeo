from commonroad_geometric.common.io_extensions.eq import monkey_patch__eq__
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle

# Monkey-patching equality operator for tests
monkey_patch__eq__()


def test_testdata_integrity(
    expected_paths_xml,
    expected_paths_pkl
):
    for xml_path, pkl_path in zip(
        expected_paths_xml,
        expected_paths_pkl,
        strict=True
    ):
        xml_scenario_bundle = ScenarioBundle.read(
            scenario_path=xml_path,
            lanelet_assignment=True
        )
        pkl_scenario_bundle = ScenarioBundle.read(
            scenario_path=pkl_path,
            lanelet_assignment=True
        )

        assert xml_scenario_bundle == pkl_scenario_bundle


def test_testdata_integrity_no_lanelet_assignment(
    expected_paths_xml,
    expected_paths_pkl
):
    for xml_path, pkl_path in zip(
        expected_paths_xml,
        expected_paths_pkl,
        strict=True
    ):
        xml_scenario_bundle = ScenarioBundle.read(
            scenario_path=xml_path,
            lanelet_assignment=False
        )
        pkl_scenario_bundle = ScenarioBundle.read(
            scenario_path=pkl_path,
            lanelet_assignment=False
        )

        assert xml_scenario_bundle == pkl_scenario_bundle
