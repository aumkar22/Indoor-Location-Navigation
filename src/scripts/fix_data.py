from typing import List, Tuple

from src.definitions import SENSORS


def line_check(line: List[str]) -> List[List[str]]:

    """
    Function to check if length of line exceeds 10 (max sensor line length). If it does,
    split the line.

    :param line: List of sensor strings (can contain multiple sensors which need to be split)
    :return: List of split sensor lists
    """

    sensor_list = []

    if len(line) <= 10:
        return [line]

    first_line, remaining_line = split_sensor_line(line)
    sensor_list.append(first_line)
    sensor_list += line_check(remaining_line)

    return sensor_list


def split_sensor_line(line: List[str]) -> Tuple[List[str], List[str]]:

    """
    Erroneous sensor lines have multiple sensors and their readings in one line. Correct sensor
    lines have their timestamps, sensor names followed by the readings. For erroneous lines,
    timestamps of following sensors are concatenated with the last sensor reading of the
    previous sensor. This function splits a sensor line into two by splitting the concatenated
    reading+timestamp.

    :param line: List of sensor strings
    :return: Returns two split lists of sensor strings
    """

    sensor_occurrence = [
        sensor_index for sensor_index, sensor in enumerate(line) if sensor in SENSORS
    ]

    second_occurrence = sensor_occurrence[1]

    ending_for_first_occurrence = line[second_occurrence - 1][:-13]
    first_line = line[: second_occurrence - 1] + [ending_for_first_occurrence]

    timestamp_for_second_line = line[second_occurrence - 1][-13:]
    second_line = [timestamp_for_second_line] + line[second_occurrence:]

    return first_line, second_line
