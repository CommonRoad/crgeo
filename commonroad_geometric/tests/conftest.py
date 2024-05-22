from pathlib import Path
from typing import List

import pytest


@pytest.fixture(scope="package")
def root_testdata_path() -> Path:
    parent_of_file = Path(__file__).parent
    return Path(parent_of_file, 'testdata')


@pytest.fixture(scope="package")
def scenario_directory_xml(root_testdata_path) -> Path:
    return root_testdata_path / 'xml'


@pytest.fixture(scope="package")
def scenario_directory_pickle(root_testdata_path) -> Path:
    return root_testdata_path / 'pkl'


@pytest.fixture(scope="package")
def expected_files() -> List[str]:
    return [
        'ARG_Carcarana-1_7_T-1',
        'DEU_Munich-1_114_0_time_steps_1000_V1_0',
        'USA_US101-26_1_T-1',
    ]


@pytest.fixture(scope="package")
def expected_paths_xml(scenario_directory_xml, expected_files) -> List[Path]:
    return [scenario_directory_xml / f"{file}.xml" for file in expected_files]


@pytest.fixture(scope="package")
def arg_carcarana_path_xml(expected_paths_xml) -> Path:
    return expected_paths_xml[0]


@pytest.fixture(scope="package")
def usa_us101_path_xml(expected_paths_xml) -> Path:
    return expected_paths_xml[2]


@pytest.fixture(scope="package")
def expected_paths_pkl(scenario_directory_pickle, expected_files) -> List[Path]:
    return [scenario_directory_pickle / f"{file}.pkl" for file in expected_files]


@pytest.fixture(scope="package")
def arg_carcarana_pkl(expected_paths_pkl) -> Path:
    return expected_paths_pkl[0]
