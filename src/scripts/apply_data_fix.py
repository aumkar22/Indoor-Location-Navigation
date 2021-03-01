"""
Script to apply data fix. Splits erroneous data lines and saves the modified text file.
"""

from src.definitions import *
from src.scripts.fix_data import line_check

sensor_files = list(TRAIN_PATH.glob(r"**/*.txt"))

for sensor_file in sensor_files:

    number_of_corrected_lines = 1
    with sensor_file.open("r+") as f:

        sensor_text = f.read()
        for sensor_line_index, sensor_data_line in enumerate(sensor_text.splitlines()):

            sensor_line_split_delimiter = sensor_data_line.split("\t")

            if len(sensor_line_split_delimiter) <= 10:
                pass
            else:
                sensor_split_lines = line_check(sensor_line_split_delimiter)

                for correct_line_index, sensor_corrected_line in enumerate(sensor_split_lines):

                    if correct_line_index == 0 and number_of_corrected_lines == 1:
                        sensor_corrected_line_joined = (
                            "\n" + "\t".join(sensor_corrected_line) + "\n"
                        )
                    else:
                        sensor_corrected_line_joined = "\t".join(sensor_corrected_line) + "\n"

                    f.write(sensor_corrected_line_joined)
                number_of_corrected_lines += 1
