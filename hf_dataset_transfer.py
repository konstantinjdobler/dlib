import filecmp
import os
import shutil
import subprocess

import fire
from tqdm import tqdm

"""
Usage:
(1) Auto-shard all large files and upload to HF Hub: 
> python hf_dataset_transfer.py shard+upload --local_dir /path/to/source/dir --hf_repo username/dataset_name

(2) Download from HF Hub and auto-unshard all files:
> python hf_dataset_transfer.py download+unshard --local_dir /path/to/target/dir --hf_repo username/dataset_name

(3) Run test to ensure sharding and unsharding works correctly:
> python hf_dataset_transfer.py test --local_dir /path/to/source/dir --override_large_files my_large_file.bin

Use the --enable_hf_transfer flag to enable HF_HUB_ENABLE_HF_TRANSFER=1 for faster up/download speed. This needs `pip install hf_transfer`.

------------------ Advanced Usage (you probably don't need this) ------------------

(4) Shard a specific file or files and upload to HF Hub:
> python hf_dataset_transfer.py shard+upload --local_dir /path/to/source/dir --hf_repo username/dataset_name --override_large_files file1.bin,file2.bin

(5) Skip sharding and upload specific files to HF Hub while ignoring large files:
> python hf_dataset_transfer.py upload --local_dir /path/to/source/dir --hf_repo username/dataset_name --override_large_files file1.bin,file2.bin

(6) Download from HF Hub and unshard specific files while re-mapping to new output file names:
> python hf_dataset_transfer.py download+unshard --local_dir /path/to/target/dir --hf_repo username/dataset_name --override_large_files file1.bin,file2.bin --override_output_files new_file1.bin,new_file2.bin

"""


def shard_large_file(source_dir, data_file="train.bin", max_size=50 * 1000**3):
    data_path = os.path.join(source_dir, data_file)
    file_size = os.path.getsize(data_path)
    num_parts = (file_size + max_size - 1) // max_size  # Calculate the number of parts

    print(f"Sharding {data_file} into {num_parts} parts...")

    with open(data_path, "rb") as f:
        for part_num in tqdm(range(num_parts), desc="Sharding", unit="part"):
            chunk = f.read(max_size)
            part_filename = f"{data_file}.part{part_num}"
            part_path = os.path.join(source_dir, part_filename)
            with open(part_path, "wb") as chunk_file:
                chunk_file.write(chunk)
            print(f"Created shard: {part_path}")


def unshard_file(target_dir, output_file="train.bin", shard_base_name="train.bin"):
    # Locate all shard files and sort them to ensure correct order
    shard_files = sorted(
        [f for f in os.listdir(target_dir) if f.startswith(shard_base_name) and f.split(".")[-1].startswith("part")],
        key=lambda x: int(x.split("part")[-1]),
    )

    if not shard_files:
        print(f"No shard files found in {target_dir} for base name {shard_base_name}.")
        return []

    output_path = os.path.join(target_dir, output_file)
    print(f"Unsharding into {output_file} from {len(shard_files)} parts...")

    with open(output_path, "wb") as outfile:
        for shard_file in tqdm(shard_files, desc="Unsharding", unit="part"):
            shard_path = os.path.join(target_dir, shard_file)
            with open(shard_path, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
            print(f"Processed shard: {shard_path}")
    return shard_files


def test_sharding_and_unsharding(source_dir, data_file="train.bin"):
    print("Running test...")

    # Run sharding
    print("Sharding the file...")
    shard_large_file(source_dir, data_file)

    # Run unsharding to 'test.bin'
    print("Unsharding the file...")
    unshard_file(source_dir, output_file="test.bin", shard_base_name=data_file)

    # Compare the original file with the reconstructed file
    original_file = os.path.join(source_dir, data_file)
    reconstructed_file = os.path.join(source_dir, "test.bin")

    if os.path.getsize(reconstructed_file) == 0:
        print(f"Test failed: {reconstructed_file} is empty.")
        return

    if filecmp.cmp(original_file, reconstructed_file, shallow=False):
        print("Test passed: The original file and the reconstructed file are identical.")
    else:
        print("Test failed: The original file and the reconstructed file are not identical.")


def upload_to_hf_hub(hf_repo, local_dir_path, exclude_files=None):
    cmd = f"huggingface-cli upload {hf_repo} {local_dir_path} --private --repo-type dataset"
    if exclude_files:
        cmd += f" --exclude {exclude_files}"
    print(f"Uploading to Hugging Face Hub: {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=os.environ)


def download_from_hf_hub(hf_repo, local_dir_path):
    cmd = f"huggingface-cli download {hf_repo} --repo-type dataset --local-dir {local_dir_path} --local-dir-use-symlinks False"
    print(f"Downloading from Hugging Face Hub: {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=os.environ)


def main(
    action,
    local_dir,  # source/target directory for data
    auto_sharding=True,
    hf_repo=None,  # if you want to up/download from/to HF Hub
    override_large_files=None,  # comma separated list of large files to shard
    override_output_files=None,  # comma separated list of output files for unsharding
    max_size=50 * 1000**3,  # 50GB limit
    enable_hf_transfer=False,  # HF_TRANSFER=1 for faster up/download speed
):
    if enable_hf_transfer:
        if not hf_repo:
            raise ValueError("hf_repo is required for Hugging Face transfer.")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if override_large_files:
        auto_sharding = False

    if action in ["shard", "upload", "shard+upload"]:
        ########### 1. Shard large file ###########
        if "shard" in action:
            existing_shards = [f for f in os.listdir(local_dir) if ".part" in f]
            if len(existing_shards) > 0:
                print(f"Warning: existing shards found: {existing_shards}, delete them? (y/n)")
                if input().lower() == "y":
                    for f in existing_shards:
                        os.remove(os.path.join(local_dir, f))
                else:
                    print("Aborting for safety to avoid overwriting existing shards, you can uncomment the code to proceed.")
                    return

            if auto_sharding:
                large_files = [f for f in os.listdir(local_dir) if os.path.getsize(os.path.join(local_dir, f)) > max_size]
                print(f"Auto-sharding large files: {large_files}")
            else:
                large_files = override_large_files.split(",")
            for file in large_files:
                shard_large_file(local_dir, file, max_size)

        ########### 2. Upload ###########
        if "upload" in action:
            if hf_repo:
                if override_large_files and not auto_sharding:
                    large_files = override_large_files.split(",")
                upload_to_hf_hub(hf_repo, local_dir, exclude_files="|".join(large_files))
            else:
                print("hf_repo is required for upload.")

    if action in ["download", "unshard", "download+unshard"]:
        ########### 1. Download ###########
        if "download" in action:
            if hf_repo:
                download_from_hf_hub(hf_repo, local_dir)
            else:
                print("hf_repo is required for download.")
        ########### 2. Unshard ###########
        if "unshard" in action:
            if auto_sharding:
                unshard_base_names = set(f.rsplit(".part", 1)[0] for f in os.listdir(local_dir) if ".part" in f)
                print(f"Auto-unsharding files: {unshard_base_names}")
            else:
                unshard_base_names = override_large_files.split(",")

            if override_output_files:
                output_files = override_output_files.split(",")
            else:
                output_files = unshard_base_names

            shards = []
            for output_file, base_name in zip(output_files, unshard_base_names):
                shards += unshard_file(local_dir, output_file=output_file, shard_base_name=base_name)

            print("Unsharding done, delete shards? (y/n)")
            if input().lower() == "y":
                for f in shards:
                    os.remove(os.path.join(local_dir, f))

    if action == "test":
        test_sharding_and_unsharding(local_dir, "train.bin")
    if action not in ["shard", "unshard", "test", "shard+upload", "download+unshard", "upload", "download"]:
        raise ValueError(
            "Invalid action. Use 'shard', 'unshard', 'test', 'shard+upload', 'download+unshard', 'upload', or 'download'."
        )


if __name__ == "__main__":
    fire.Fire(main)
