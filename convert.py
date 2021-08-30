# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import argparse


def replace_module_prefix(state_dict, prefix, replace_with=""):
    """
    Remove prefixes in a state_dict needed when loading models that are not VISSL
    trained models.
    Specify the prefix in the keys that should be removed.
    """
    state_dict = {
        (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def convert_and_save_model(args, replace_prefix):
    model_path = args.model_path
    model = torch.load(model_path, map_location=torch.device("cpu"))

    # get the model trunk to rename
    if "classy_state_dict" in model.keys():
        model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in model.keys():
        model_trunk = model["model_state_dict"]
    else:
        model_trunk = model
    print(f"Input model loaded. Number of params: {len(model_trunk.keys())}")

    # convert the trunk
    converted_model = replace_module_prefix(model_trunk, "_feature_blocks.")
    print(f"Converted model. Number of params: {len(converted_model.keys())}")

    # save the state
    output_filename = f"{args.output_name}.pth"
    output_model_filepath = f"{args.output_dir}/{output_filename}"
    print(f"Saving model: {output_model_filepath}")
    torch.save(converted_model, output_model_filepath)
    print("DONE!")
    print(f"Input model : {model_path}")
    print(f"Output model : {output_model_filepath}")

    return converted_model


def compare_keys(source_dict, target_dict_path):
    print("\n Comparing keys \n")
    target_dict = torch.load(target_dict_path, map_location=torch.device("cpu"))
    if 'state_dict' in target_dict.keys():
        target_dict = target_dict['state_dict']

    print(f"Same number of params : {len(source_dict.keys()) == len(target_dict.keys())} \
         Source/Target = {len(source_dict.keys())}/{len(target_dict.keys())}")

    for key in source_dict.keys():
        if key not in target_dict.keys():
            print(f"{key} in source dict is not in target dict.")

    for key in target_dict.keys():
        if key not in source_dict.keys():
            print(f"Source dict doesn't have {key}.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VISSL ResNe(X)ts models to Torchvision"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Model url or file that contains the state dict",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory where the converted state dictionary will be saved",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        required=True,
        help="output model name"
    )
    parser.add_argument(
        "--target_dict_path",
        type=str,
        default=None,
        required=True,
        help="Compare keys of converted model dict and target dict"
    )
    args = parser.parse_args()
    converted_model = convert_and_save_model(args, replace_prefix="_feature_blocks.")
    if args.target_dict_path is not None:
        compare_keys(converted_model, args.target_dict_path)


if __name__ == "__main__":
    main()