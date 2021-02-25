from typing import List, Union

from src.definitions import *


def sensor_count_check(line: List[str]):

    return dict((sensor, line.count(sensor)) for sensor in set(line) if sensor in SENSORS)


def line_check(line):

    if len(line) == 2:
        line_to_split = line[-1]
        sensor_count = sensor_count_check(line_to_split)

    else:
        sensor_count_check(line)


def line_separator_fix(line: Union[List[List[str]], List[str]]):

    if len(line) == 2:
        line_to_split = line[-1]
        sensor_count = sensor_count_check(line_to_split)
        breakpoint()
        if (
            len(sensor_count) == 1
            and len(line_to_split) <= 10
            and list(sensor_count.values())[0] == 1
        ):
            return "\t".join(line_to_split)

        elif len(sensor_count.values()) > 1:
            first_occurrence = line_to_split.index(list(sensor_count.keys())[0], 0)
            second_occurrence = line_to_split.index(
                list(sensor_count.keys())[0], first_occurrence + 1
            )

            ending_for_first_occurrence = line_to_split[second_occurrence - 1][:-13]
            first_line = line_to_split[: second_occurrence - 1] + [ending_for_first_occurrence]

            timestamp_for_second_occurrence = line_to_split[second_occurrence - 1][-13:]
            split_line = [timestamp_for_second_occurrence] + line_to_split[second_occurrence:]

            breakpoint()
            return line_separator_fix([first_line, split_line])
