import pytest

from finsight.tools import CalculationError, execute_program


def test_nested_percentage_program():
    result = execute_program("multiply(divide(subtract(562,491),491),const_100)")
    assert result.value == pytest.approx(14.4602851324)
    assert len(result.steps) == 3


def test_finqa_constant_and_divide():
    assert execute_program("divide(637, const_5)").value == pytest.approx(127.4)


def test_rejects_arbitrary_python():
    with pytest.raises(CalculationError):
        execute_program("__import__('os').system('echo unsafe')")


def test_rejects_division_by_zero():
    with pytest.raises(CalculationError):
        execute_program("divide(1, 0)")


def test_multistep_references_and_percent_literal():
    result = execute_program("subtract(959.2, 991.1), divide(#0, 991.1)")
    assert result.value == pytest.approx((959.2 - 991.1) / 991.1)
    assert execute_program("divide(9896, 23.6%)").value == pytest.approx(9896 / 0.236)


def test_table_operation_uses_named_row():
    table = [["year", "2014", "2015"], ["beginning balance", "$ 10", "20"]]
    result = execute_program("table_average(beginning balance, none)", table=table)
    assert result.value == pytest.approx(15)


def test_table_operation_ignores_second_arg_like_official_evaluator():
    table = [["for", "against", "abstained"], ["17695228", "963202", "155213"]]
    result = execute_program("table_sum(17695228, 963202)", table=table)
    assert result.value == pytest.approx(1_118_415)


def test_duplicate_table_labels_use_last_row_like_official_evaluator():
    table = [["year", "high", "low"], ["first quarter", "62", "54"], ["first quarter", "38", "34"]]
    result = execute_program("table_max(first quarter, none)", table=table)
    assert result.value == pytest.approx(38)


def test_greater_matches_finqa_semantics():
    assert execute_program("greater(45, 25)").value == "yes"
