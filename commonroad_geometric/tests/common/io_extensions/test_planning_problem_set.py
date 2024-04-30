from commonroad_geometric.common.io_extensions.hash import hash_goal, hash_planning_problem, hash_planning_problem_set


def test_hash_planning_problem_set_deterministic(usa_us101_planning_problem_set_xml):
    hash1 = hash_planning_problem_set(usa_us101_planning_problem_set_xml)
    hash2 = hash_planning_problem_set(usa_us101_planning_problem_set_xml)
    assert hash1 == hash2


def test_hash_planning_problem_set_changes_on_change(
    usa_us101_planning_problem_set_xml,
    usa_us101_planning_problem_xml
):
    old_hash = hash_planning_problem_set(usa_us101_planning_problem_set_xml)
    usa_us101_planning_problem_xml._planning_problem_id = 1
    usa_us101_planning_problem_set_xml.add_planning_problem(usa_us101_planning_problem_xml)
    new_hash = hash_planning_problem_set(usa_us101_planning_problem_set_xml)
    assert old_hash != new_hash


def test_hash_planning_problem_deterministic(usa_us101_planning_problem_xml):
    hash1 = hash_planning_problem(usa_us101_planning_problem_xml)
    hash2 = hash_planning_problem(usa_us101_planning_problem_xml)
    assert hash1 == hash2


def test_hash_planning_problem_changes_on_change(usa_us101_planning_problem_xml):
    old_hash = hash_planning_problem(usa_us101_planning_problem_xml)
    usa_us101_planning_problem_xml._planning_problem_id = usa_us101_planning_problem_xml.planning_problem_id + 1
    new_hash = hash_planning_problem(usa_us101_planning_problem_xml)
    assert old_hash != new_hash


def test_hash_goal_deterministic(usa_us101_planning_problem_xml):
    hash1 = hash_goal(usa_us101_planning_problem_xml.goal)
    hash2 = hash_goal(usa_us101_planning_problem_xml.goal)
    assert hash1 == hash2


def test_hash_goal_changes_on_change(usa_us101_planning_problem_xml):
    old_hash = hash_goal(usa_us101_planning_problem_xml.goal)
    usa_us101_planning_problem_xml.goal.state_list.clear()
    new_hash = hash_goal(usa_us101_planning_problem_xml.goal)
    assert old_hash != new_hash
