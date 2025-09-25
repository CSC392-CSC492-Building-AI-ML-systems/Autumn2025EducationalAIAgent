import re
import argparse
from pathlib import Path


def chunk_input(filepath):
    with open(filepath) as file:
        input = file.read()

    pattern = re.compile(
        r"<(?:user_input|system_output)\b[^>]*?(?:/>|>.*?</(?:user_input|system_output)>)",
        flags=re.DOTALL,
    )
    chunks = pattern.findall(input)

    return chunks


def get_output(filepath):
    with open(filepath) as file:
        output = file.readlines()

    return output


def group_output_xml_txt(input_filepath, output_filepath):
    chunks = chunk_input(input_filepath)
    output = get_output(output_filepath)

    # curr_group = []
    chunk_tuples = []
    current_group = 0
    curr_line = 2
    for chunk in chunks:
        lines = len(chunk.splitlines())
        curr_line += lines

        chunk_tuples.append((chunk, current_group))
        if output[curr_line] == "0\n":
            current_group += 1

    return chunk_tuples


def output_to_csv(lst, file_number, output_dir):
    output_path = Path(output_dir).joinpath(file_number + ".csv")
    with open(output_path, "w+") as file:
        for val in lst:
            file.write(val[0] + "," + str(val[1]) + "\n")


def get_base_filename(filepath):
    return Path(filepath).stem.split(".")[0]


def main():
    # parser = argparse.ArgumentParser(
    #     description="Convert old data format to new format"
    # )
    # parser.add_argument("--xml", help="Path to XML input file")
    # parser.add_argument("--output", help="Path to output file")
    #
    # parser.add_argument(
    #     "--output-dir",
    #     default="new_data",
    #     help="Directory for new streaming format files",
    # )
    #
    # args = parser.parse_args()
    #
    # if not args.xml or not args.output:
    #     print("Error: --xml or --output missing")
    #
    # if not args.output_dir:
    #     print("Error: --output_dir missing")
    #     return

    input_file = "../../../data/model_0/inputs/1717168649.rec.xml"
    output_file = "../../../data/model_0/outputs/1717168649.xml.txt"
    output_dir = "../../../data/model_0/new_format/"

    chunks = group_output_xml_txt(input_file, output_file)
    file_number = get_base_filename(input_file)
    output_to_csv(chunks, file_number, output_dir)
    print("Converted old input + output files into the new training format.")


if __name__ == "__main__":
    main()
