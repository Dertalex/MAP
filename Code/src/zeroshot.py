import pyrosetta
rosetta.init()("mute all")
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta import create_score_function


def calculate_delta_G(pose, scorefxn=pyrosetta.get_fa_scorefxn()):
    """
    Calculate the delta G of a pose using the provided score function.
    If no score function is provided, the default score function is used.
    """
    if scorefxn is None:
        scorefxn = pyrosetta.get_fa_scorefxn()   
    return scorefxn(pose)

def mutate_pose(pose, position, new_residue):
    """
    Mutate a pose at a specific position to a new residue.
    """
    mutant_pose = pose.clone()
    mutate_residue(mutant_pose, position, new_residue)
    return mutant_pose  

def prepare_pose(pose, scorefxn):
    # pack sidechains around mutation site
    task = pyrosetta.standard_packer_task(pose)
    task.restrict_to_repacking()
    task.or_include_current(True)
    packer = PackRotamersMover(scorefxn, task)
    packer.apply(pose)
    
    # relax the pose
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)

    # minimize the pose
    move_map = MoveMap()
    movemap.set_bb(True)  # Allow backbone movements
    movemap.set_chi(True)  # Allow sidechain movements
    
    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn)
    min_mover.apply(pose)

    return pose


def zero_shot_mutants(pdb, mutants_list, scorefxn=create_score_function("ref15")):
    """
    Generate zero-shot mutants for a given PDB file.
    
    Args:
        pdb_path (str)
        mutants_list: format e.g.: [A234F,...]
        scorefxn: Optional; a custom score function to use for energy calculations.
    
    Returns:
        list: A list of tuples containing the mutant and their delta delta G.
        possible scoring functions:
            =create_score_function("ref15")
            =create_score_function("cen_std")
            =pyrosetta.get_fa_scorefxn()
    """
    

    pose_wt = pose_from_pdb(pdb_path)
    prepared_wildtype = prepare_pose(pose_wt)
    deltaG_wd = calculate_delta_G(prepared_wildtype, scorefxn)
	mutant_list = []

    for mutant in mutants_list:
    	position = int(mutant[1:-1])
    	new_residue = mutant[-1]
        pose_mut = mutate_pose(prepared_wildtype, position, new_residue)
		prepared_mutant = prepare_pose(pose_mut)
    	deltaG_mut = calculate_delta_G(prepared_mutant, scorefxn)

    	mutant_list.append((mutant, deltaG_mut - deltaG_wd))

    return mutant_list
	
