# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from scripts.convert_pkl_to_chgcar import deepdft_to_chgcar
import argparse
from pathlib import Path
import glob
import tqdm
from multiprocessing.pool import Pool


parser = argparse.ArgumentParser()
parser.add_argument('npy_dir', help='Directory with cube files in .npy format with predicted charge and/or spin densities.')
parser.add_argument('chgcar_dir', help='Directory of original CHGCARs')
parser.add_argument('out_dir', help='Directory to output CHGCAR files. Will put a CHGCAR in individual folders.')
parser.add_argument("--workers", type=int, default=1, help="Number of workers to run conversion")

def convert(npy_file, atoms_file, output_file, aug_file=None):
    '''Convert and write a single set of files'''
    chgcar = deepdft_to_chgcar(npy_file, atoms_file, aug_chgcar_file=aug_file)
    chgcar.write(output_file, format="chgcar")

def main(npy_dir: Path, out_dir: Path, aug_chgcar_dir: Path = None, workers: int = 1):
    '''Convert and write all of the files in the specified directories'''
    npys = [str(Path(i)) for i in glob.glob(str(npy_dir / '*.npy'))]
    pkls = [p[:-3] + "_atoms.pkl" for p in npys]
    mpids = [i.stem for i in npys]
    outs = [str(out_dir / f"{mpid}.CHGCAR") for mpid in mpids]
    if aug_chgcar_dir is not None:
        augs = [str(aug_chgcar_dir / mpid / "CHGCAR") for mpid in mpids] 
    else:
        augs = [None]*len(outs)
    
    if workers <= 1:
        for f_npy, f_pkl, f_out in tqdm(zip(npys, pkls, outs, augs)):
            convert(f_npy, f_pkl, f_out)
    else:
        with Pool(workers) as p:
            p.starmap(convert, zip(npys, pkls, outs, augs))
    
if __name__ == "__main__":
    args = parser.parse_args()
    npy_dir = Path(args.npy_dir)
    chgcar_dir = Path(args.chgcar_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    main(npy_dir, chgcar_dir, out_dir, workers=args.workers)