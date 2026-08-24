import argparse
from tqdm import tqdm

from src.utils import load_env, load_json, save_to_json, get_logger

def main(args):

    logger = get_logger("remove_null_res")

    # open the file
    full_data = load_json(args.input_file, logger)
    to_clean = full_data["data"]

    # clean the file
    clean_data = {}
    for k, v in tqdm(to_clean.items(), desc="cleaning data"):
        if v["annotation"] is not None:
            clean_data[k] = v

    # overwrite
    full_data["data"] = clean_data
    out_path = ("/").join(args.input_file.split("/")[:-1])
    file_name = args.input_file.split("/")[-1]
    save_to_json(full_data, out_path, file_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input file path')
    args = parser.parse_args()
    main(args)
