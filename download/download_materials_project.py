# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from multiprocessing import pool
from pathlib import Path
import argparse
import json
from mp_api.client import MPRester
from emmet.core.summary import HasProps
from emmet.core.tasks import TaskDoc
from pymatgen.io.vasp import Chgcar
from typing import Tuple, Union



parser = argparse.ArgumentParser()
parser.add_argument("--out_path", type=str, required=True,
    help="Top level directory to save outputs to")
parser.add_argument("--limit", type=int, default=0, 
    help="Limit number of mpids to process (for debugging)")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--task_id_file", type=str, help="(optional) json file mapping from `material_id`s to `task_id`s")
parser.add_argument("--mp_api_key", type=str, help="API Key from materials project")


def get_charge_density_with_task_docs(MP_API_KEY: str, mpid: str, deserialize: bool = False) -> Tuple[Chgcar, TaskDoc]:
    r'''Retrieve charge density and the associated TaskDoc
    
    See https://github.com/materialsproject/api/issues/761 if there are problems.

    The TaskDoc returned is the one associated with calculations of 
    charge density. Note that the task_id for this TaskDoc may not match the 
    material_id that was queried. An explanation is offered here:

    https://docs.materialsproject.org/frequently-asked-questions#what-is-a-task_id-and-what-is-a-material_id-and-how-do-they-differ

    TaskDocs cannot be deserialized into the pymatgen objects that represent
    the VASP input files without VASP pseudopotentials. If these pseudopotentials
    are available, pymatgen can be configured to use them to run deserialization:

    https://pymatgen.org/installation.html#potcar-setup

    '''
    with MPRester(MP_API_KEY, monty_decode=deserialize) as mpr:
        chgcar, taskdoc = mpr.get_charge_density_from_material_id(mpid, inc_task_doc=True)
    return chgcar, taskdoc


def get_charge_density_and_task_docs_by_task_id(MP_API_KEY: str, task_id: str, deserialize: bool = False) -> Tuple[Chgcar, TaskDoc]:
    r'''Retrieve charge density and the associated TaskDoc, from the task_id

        This requires a few more steps than get_charge_density_with_task_docs, since this is not supported
        with the existing mp_api package
    
    '''
    with MPRester(MP_API_KEY, monty_decode=deserialize) as mpr:
        task_doc = mpr.tasks.get_data_by_id(task_id)
        results = mpr.charge_density.search(task_ids=[task_id])  # type: ignore
        assert len(results) == 1
        chgcar = mpr.charge_density.get_charge_density_from_file_id(results[0].fs_id)
        return chgcar, task_doc


def get_materials_with_charge_density(MP_API_KEY: str) -> list:
    r'''Returns a list of material ids that have charge density data'''
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.summary.search(has_props = [HasProps.charge_density], fields=["material_id"])
    charge_density_mpids = [doc.material_id for doc in docs]
    return charge_density_mpids

def write_chgcar(chgcar: Chgcar, root_dir: Union[Path, str] , mpid: str):
    r'''Write CHGCAR to file'''
    write_dir = Path(root_dir) / mpid
    write_dir.mkdir(parents=True, exist_ok=True)
    chgcar.write_file(str(write_dir / "CHGCAR"))
    
    
def write_task_id_to_txt(taskdoc: TaskDoc, root_dir: Union[Path, str], mpid: str):
    r'''Write task_id to txt file for tracking'''
    fname = Path(root_dir) / mpid / "task_id.txt"
    with open(fname, "w") as f:
        f.write(f"{taskdoc.task_id}\n")
        
        
def write_filelist(mpids: list, filename: Union[str, Path]):
    with open(filename, 'w') as f:
        for mpid in mpids:
            f.write(f"{mpid}\n")


def read_filelist(filename: Union[str, Path]) -> list:
    with open(filename) as file:
        mpids = [mpid.rstrip() for mpid in file]
    return mpids


def _read_in_write_out(mp_api_key: str, mpid: str, outpath: Union[Path, str]):
    outpath=Path(outpath)
    try:
        chgcar, taskdoc = get_charge_density_with_task_docs(mp_api_key, mpid, deserialize=False)
    except Exception as e:
        print(e)
        return

    if chgcar is not None:
        write_chgcar(chgcar, outpath, mpid)
        
    if taskdoc is not None:
        # This will just record the taskID, NOT decode and write task docs
        write_task_id_to_txt(taskdoc=taskdoc, root_dir=outpath, mpid=mpid)
        
        
def _read_in_write_out_task(mp_api_key: str, mpid: str, task_id: str, outpath: Union[Path, str]):
    outpath=Path(outpath)
    try:
        chgcar, taskdoc = get_charge_density_and_task_docs_by_task_id(mp_api_key, task_id, deserialize=False)
    except Exception as e:
        print(e)
        return

    if chgcar is not None:
        write_chgcar(chgcar, outpath, mpid)
    else:
        print(f"chgcar for {mpid}, {task_id}, not returned!")
        
    if taskdoc is not None:
        # This will just record the taskID, NOT decode and write task docs
        write_task_id_to_txt(taskdoc=taskdoc, root_dir=outpath, mpid=mpid)
    else:
        print(f"task_doc for {mpid}, {task_id}, not returned!")


def main(args):    
    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    
    ids_file = out_path / "filelist.txt"

    
    if args.task_id_file is not None:
        # Use the json file mapping mpid to task_id to download the CHGCARs from specific tasks
        print("Gathering material ids and task ids from existing file")
        with open(args.task_id_file, "r") as f:
            items = json.load(f)
        
        mpids = [k for k in items.keys()]
        task_ids = [v for v in items.values()]
        
        if args.limit > 0:
            print(f"Limiting to {args.limit} materials.")
            mpids = mpids[:args.limit]
            task_ids = task_ids[:args.limit]

        if args.workers > 5:
            raise ValueError("Num workers should be less than 5, to avoid strain on MP servers") 
        elif args.workers > 1:
            print(f"Starting API calls with {args.workers} workers...")
            pool.Pool(args.workers).starmap(_read_in_write_out_task, [(args.mp_api_key, mpid, task_id, args.out_path) for mpid, task_id in zip(mpids, task_ids)])
        else:
            print("Starting API call with a single worker. Add additional workers with --workers")
            for mpid, task_id in zip(mpids, task_ids):
                _read_in_write_out_task(args.mp_api_key, mpid, task_id, args.out_path)
            
            
    else:
        print("Retrieve SummaryDocs to find materials with charge density data....")
        mpids = get_materials_with_charge_density(args.mp_api_key)
        write_filelist(mpids, ids_file)
        mpids = read_filelist(ids_file)
        
        if args.limit > 0:
            print(f"Limiting to {args.limit} materials.")
            mpids = mpids[:args.limit]
        
        if args.workers > 5:
            raise ValueError("Num workers should be less than 5, to avoid strain on MP servers") 
        elif args.workers > 1:
            print(f"Starting API calls with {args.workers} workers...")
            pool.Pool(args.workers).starmap(_read_in_write_out, [(args.mp_api_key, mpid, args.out_path) for mpid in mpids])
        else:
            print("Starting API call with a single worker. Add additional workers with --workers")
            for mpid in mpids:
                _read_in_write_out(args.mp_api_key, mpid, args.out_path)
    
    print("Script Completed.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
