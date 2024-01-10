# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from pathlib import Path
from scripts.convert_chgcar_to_pkl import convert as chgcar_to_npypkl
from scripts.write_mp_probe_count_file import count_elements_in_numpy_files
from scripts.write_dummy_split import main as write_split
from tqdm import tqdm
import argparse
from functools import partial
from multiprocessing.pool import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Directory containing **/CHGCAR files")
parser.add_argument("--output", type=str, help="Directory to save .npy and .pkl files to")
parser.add_argument("--spin", action="store_true", help="If added, .npy files will be written with spin information")
parser.add_argument("--workers", type=int, default=1, help="Number of workers to run conversion")

def convert(f, input_dir=None, output_dir=None, write_filelist=True, write_spin=False):
    
    relative_path_parts = f.relative_to(input_dir).parts
    print(relative_path_parts)
    fname = "_".join(relative_path_parts[:-1]) # exclude CHGCAR part
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    filelist = output_dir / "filelist.txt" if write_filelist else None
    
    chgcar_to_npypkl(f, 
                    output_dir / f"{fname}.npy", 
                    output_dir / f"{fname}_atoms.pkl",
                    filelist,
                    overwrite=True,
                    spin=write_spin
                     )



if __name__ == "__main__":
    args = parser.parse_args()
    perturbed_structures_dir = Path(args.input)
    output_dir = Path(args.output)
    workers = int(args.workers)
    write_filelist = True
    
    _convert = partial(
        convert, 
        input_dir=perturbed_structures_dir, 
        output_dir=output_dir, 
        write_filelist=write_filelist,
        write_spin=args.spin
    )
    
    # find all chgcar files in directory
    chgcar_files = perturbed_structures_dir.rglob("**/CHGCAR")
    
    filelist_file = output_dir / "filelist.txt"
    split_file = output_dir / "split.json"
    
    # convert each file to deepdft inputs and place in a special directory
    if workers <= 1:
        for f in tqdm(chgcar_files):
            _convert(f)
        count_elements_in_numpy_files(file_list_path=filelist_file, workers=1)
    else:
        with Pool(workers) as p:
            p.map(_convert, chgcar_files)
        count_elements_in_numpy_files(file_list_path=filelist_file, workers=workers)
    
    write_split(filelist_file, output_file=split_file)
