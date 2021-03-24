from src.util.definitions import *
from src.scripts.fix_data import line_check


def fix_data_issues():

    """
    Script to apply data fix. Splits erroneous data lines and saves the modified text file.
    """

    for sensor_file in sensor_train_files:
        number_of_corrected_lines = 1
        with sensor_file.open("r+", encoding="utf8") as f:

            sensor_text = f.read()
            for sensor_line_index, sensor_data_line in enumerate(sensor_text.splitlines()):

                sensor_line_split_delimiter = sensor_data_line.split("\t")

                if len(sensor_line_split_delimiter) <= 10:
                    pass
                else:
                    sensor_split_lines = line_check(sensor_line_split_delimiter)
                    print("Error lines found in file: ", sensor_file)
                    print("# error lines", len(sensor_split_lines))
                    for correct_line_index, sensor_corrected_line in enumerate(sensor_split_lines):

                        if correct_line_index == 0 and number_of_corrected_lines == 1:
                            sensor_corrected_line_joined = (
                                "\n" + "\t".join(sensor_corrected_line) + "\n"
                            )
                        else:
                            sensor_corrected_line_joined = "\t".join(sensor_corrected_line) + "\n"

                        f.write(sensor_corrected_line_joined)

                    print("Fixed error line: ", sensor_line_index)
                    number_of_corrected_lines += 1
