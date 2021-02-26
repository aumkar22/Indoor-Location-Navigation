from typing import List, Tuple, Dict, Optional

from src.definitions import *


def sensor_count_check(line: List[str]):

    return dict((sensor, line.count(sensor)) for sensor in set(line) if sensor in SENSORS)


def append_fixed_line(line: List[str], list_of_sensors: List = []):

    list_of_sensors.append(line)
    return list_of_sensors


def line_check(line: List[str], fixed_sensor_list: Optional):

    sensor_count = sensor_count_check(line)

    if len(sensor_count) > 1 or len(sensor_count.values()) > 1:
        first_line, remaining_line = line_separator_fix(sensor_count, line)
        sensor_list = append_fixed_line(first_line)

        if fixed_sensor_list is not None:
            sensor_list.append(fixed_sensor_list)

        return line_check(remaining_line, sensor_list)
    else:
        sensor_list = append_fixed_line(line)
        return sensor_list


def split_sensor_line(line: List[str], sensor_occurrence: int):

    ending_for_first_occurrence = line[sensor_occurrence - 1][:-13]
    first_line = line[: sensor_occurrence - 1] + [ending_for_first_occurrence]

    timestamp_for_second_occurrence = line[sensor_occurrence - 1][-13:]
    second_line = [timestamp_for_second_occurrence] + line[sensor_occurrence:]

    return [first_line, second_line]


def line_separator_fix(sensor_count: Dict, line: List[str]) -> Tuple[List[str], List[str]]:

    first_occurrence = line.index(list(sensor_count.keys())[0], 0)

    try:
        second_occurrence = line.index(list(sensor_count.keys())[0], first_occurrence + 1)
        first_line, split_line = split_sensor_line(line, second_occurrence)

    except ValueError:
        split_line, first_line = split_sensor_line(line, first_occurrence)

    return first_line, split_line
