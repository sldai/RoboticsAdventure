import numpy as np

def preprocess_data(odom_fname, laser_fname):
    """Read the odom and laser data

    Args:
        odom_fname
        laser_fname

    Returns:
        data_dict: odom data: each line contain ranges
                   scan data: each line contain [dx,dy,dtheta]

    """
    odom = []
    for line in open(odom_fname, "r"):
        line = line.rstrip()
        strs = line.split(" ")
        odom.append(list(map(float,strs)))
    scan = []
    for line in open(laser_fname, "r"):
        line = line.rstrip()
        strs = line.split(" ")
        scan.append(list(map(float,strs)))    
    assert len(odom)==len(scan)
    data_dict = {'odom': np.array(odom), 'scan': np.array(scan)} 
    return data_dict



