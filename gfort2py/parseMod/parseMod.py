# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
import gzip
import os
import pickle
import sys
import subprocess

PYFILE_VERSION = 2

class VersionError(Exception):
    pass


def loadData(filename):
    try:
        with gzip.open(filename) as f:
            x = f.read()
    except OSError as e:
        e.args = [filename + " is not a valid .mod file"]
        raise
    return parseInput(x, filename)

def hashFile(filename):
    p = subprocess.check_output(["md5sum", filename])
    return p.decode().split()[0]

def parseInput(x, filename):
    res = {}
    
    x = x.decode()
    header = x.split('\n')[0]
    x = x.replace('\n', ' ')
    
    if 'GFORTRAN' not in header:
            raise ValueError('Not a gfortran mod file')
    
    res['version'] = int(header.split("'")[1])
    res['orig_file'] = header.split()[-1]
    res['filename'] = filename
    res['checksum'] = hashFile(filename)
    
    if res['version'] == 14:
            from . import parseMod14 as p
    elif res['version'] == 15:
            from . import parseMod15 as p
    else:
            raise VersionError("Only supports mod version 14")
    
    pm = p.parseMod(x,filename,res,PYFILE_VERSION)
    
    return pm
	
def run(filename,output=None,save=True,unpack=True):
    x = loadData(filename)
    x.processData()
    if save:
        if output is None:
            output=fpyname(filename)
        
        x.save(output)
        
    if unpack:
        return x.getUnpackedData()
    else:
        return x

def fpyname(filename):
    return filename.split('.')[0] + '.fpy'

#################################

if __name__ == "__main__":
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
        
    for filename in files:
        run(filename,save=True,unpack=False)
