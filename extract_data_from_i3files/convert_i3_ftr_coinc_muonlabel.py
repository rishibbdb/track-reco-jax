from icecube import dataio, hdfwriter, icetray, MuonGun, dataclasses
from icecube.sim_services.label_events import MCLabeler, MuonLabels
from icecube.sim_services.label_events import ClassificationConverter
from icecube.icetray import I3Tray
from icecube.dataclasses import I3Particle
from icecube.icetray import I3Units
from icecube.icetray import *
from icecube.sim_services.label_events.enums import classification
import numpy as np
import warnings

import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.stats import truncnorm
import pandas as pd
import os
import glob
# os.path.append('/mnt/home/baburish/jax/TriplePandelReco_JAX')
from _lib.pulse_extraction_from_i3 import get_pulse_info

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-id", "--indir", type=str,
                  default="/home/storage2/hans/i3files/21217/",
                  dest="INDIR",
                  help="directory containing the .i3 files")

parser.add_argument("-ib", "--infile_base", type=str,
                  default="wBDT_wDNN_L345_IC86-2016_NuMu",
                  dest="INFILE_BASE",
                  help="part of filename that is common to all .i3 files")

parser.add_argument("-is", "--infile_suffix", type=str,
                  default=".i3.zst",
                  dest="INFILE_SUFFIX",
                  help="suffix of .i3 files. Typically .i3.zst")

parser.add_argument("-did", "--dataset_id", type=int,
                  default=21217,
                  dest="DATASET_ID",
                  help="ID of IceCube dataset")

parser.add_argument("-s", "--file_index_start", type=int,
                  default=10000,
                  dest="FILE_INDEX_START",
                  help="start index of range of files to be converted (included)")

parser.add_argument("-e", "--file_index_end", type=int,
                  default=20000,
                  dest="FILE_INDEX_END",
                  help="end index of range of files to be converted (excluded)")

parser.add_argument("-o", "--outdir", type=str,
                  default="/home/storage2/hans/i3files/21217/",
                  dest="OUTDIR",
                  help="directory where to write output feather files")

parser.add_argument('--recompute_true_muon_energy', action='store_true',
                  dest="RECOMPUTE_MU_E")

args = parser.parse_args()

dataset_id = args.DATASET_ID
indir = args.INDIR
infile_base = args.INFILE_BASE
infile_suffix = args.INFILE_SUFFIX
file_index_start = args.FILE_INDEX_START
file_index_end = args.FILE_INDEX_END
outdir = args.OUTDIR

if args.RECOMPUTE_MU_E:
    from _lib.muon_energy import add_muon_energy

n_events_per_file = int(1.e5)
event_count = 0
event_first_pulse_idx = 0 # inclusive
event_last_pulse_idx = 0 # inclusive
meta_keys = dict()
meta_frames = []
pulse_frames = []
meta_keys['pulses'] = 'TWSRTHVInIcePulsesIC'
meta_keys['mc_primary_neutrino'] = 'MCPrimary1'
meta_keys['mc_most_energetic_muon'] = 'MCMostEnergeticTrack'
meta_keys['spline_mpe'] = 'SplineMPEIC'
meta_keys['mc_muon_energy_at_interaction'] = 'TrueMuonEnergyAtInteraction'
meta_keys['mc_muon_energy_at_detector_entry']  = 'TrueMuoneEnergyAtDetectorEntry'
meta_keys['mc_muon_energy_at_detector_leave'] = 'TrueMuoneEnergyAtDetectorLeave'
meta_keys['bkg_mc_tree'] = 'I3MCTree'
n_events_per_file = int(1.e5)
event_count = 0
event_first_pulse_idx = 0 # inclusive
event_last_pulse_idx = 0 # inclusive
pulse_data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[]}

meta_data = {'event_id': [], 'idx_start': [], 'idx_end': [], 'n_channel_HLC': []}
meta_data.update({'neutrino_energy': [], 'muon_energy': [], 'muon_energy_at_detector': []})
meta_data.update({'muon_energy_lost': [], 'q_tot': [], 'n_channel': []})
meta_data.update({'muon_zenith': [], 'muon_azimuth': [], 'muon_time': []})
meta_data.update({'muon_pos_x': [], 'muon_pos_y': [], 'muon_pos_z': []})
meta_data.update({'spline_mpe_zenith': [], 'spline_mpe_azimuth': [], 'spline_mpe_time': []})
meta_data.update({'spline_mpe_pos_x': [], 'spline_mpe_pos_y': [], 'spline_mpe_pos_z': []})
min_muon_energy_at_detector = 1000 # GeV
max_muon_energy_at_detector = 10000 # GeV

muon_label_list = []
def muonframe(frame):
  global muon_label_list
  try:
    seed = frame['coincident_muons'].value
    if seed > 0:
      False
    else:
      True
      muon_label_list.append(seed)
  except:
    print("Error")

def framestats(frame):
  global muon_label_list
  global event_header
  global interaction_type
  global most_energetic_track
  global primary_neutrino
  global spline_mpe
  global muon_energy_at_interaction
  global muon_energy_at_det
  global muon_energy_leaving
  global meta_frames
  global pulse_frames
  global event_count

  if args.RECOMPUTE_MU_E:
            # Compute true properties of muon.
            #print("recomputing muon energy.")
            add_muon_energy(frame)
  try:
      muon_energy_at_interaction = frame[meta_keys['mc_muon_energy_at_interaction']].value # I3Double
      muon_energy_at_det =  frame[meta_keys['mc_muon_energy_at_detector_entry']].value # I3Double
      muon_energy_leaving = frame[meta_keys['mc_muon_energy_at_detector_leave']].value # I3Double
  except:
      print("Missing a key. Skip!")

  try:
    seed = frame['coincident_muons']
    event_header = frame['I3EventHeader']
    interaction_type = frame['I3MCWeightDict']['InteractionType']
    most_energetic_track = frame[meta_keys['mc_most_energetic_muon']] # I3Particle
    primary_neutrino = frame[meta_keys['mc_primary_neutrino']] # I3Particle
    spline_mpe = frame[meta_keys['spline_mpe']]
    muon_label_list.append(seed)
    # print("Loaded")
  except:
    print("Error")

  is_CC_interaction = interaction_type < 1.5
  pass_muon_energy = np.isfinite(muon_energy_at_det) and muon_energy_at_det > min_muon_energy_at_detector and muon_energy_at_det < max_muon_energy_at_detector
  energy_ratio = muon_energy_at_interaction  / most_energetic_track.energy
  found_correct_muon = energy_ratio < 0.9 or energy_ratio > 0.9
  has_sensible_muon = np.logical_and(pass_muon_energy, energy_ratio)
  if meta_keys['bkg_mc_tree'] == 'I3MCTree':
     has_no_coinc = len(frame['I3MCTree'].get_primaries()) == 1
  else:
     has_no_coinc = len(frame[meta_keys['bkg_mc_tree']]) == 0
  has_sensible_muon = np.logical_and(has_sensible_muon, has_no_coinc)
  if np.logical_and(is_CC_interaction, has_sensible_muon):
    # Retain event.
    event_count += 1

    event_id = event_header.run_id * n_events_per_file + event_header.event_id

    # Get all pulses.
    event_pulse_data, summary = get_pulse_info(frame, event_id, pulses_key=meta_keys['pulses'])
    # Store.
    for key in pulse_data.keys():
        pulse_data[key] += event_pulse_data[key]

    # Get meta_data.
    event_last_pulse_idx = event_first_pulse_idx + summary['n_pulses'] - 1
    meta_data['event_id'].append(event_id)
    meta_data['idx_start'].append(event_first_pulse_idx)
    meta_data['idx_end'].append(event_last_pulse_idx)

    meta_data['neutrino_energy'].append(primary_neutrino.energy)
    meta_data['muon_energy'].append(muon_energy_at_interaction)
    meta_data['muon_energy_at_detector'].append(muon_energy_at_det)

    if np.isfinite(muon_energy_leaving):
        meta_data['muon_energy_lost'].append(muon_energy_at_det - muon_energy_leaving)
    else:
        # lost all energy inside the detector
        meta_data['muon_energy_lost'].append(muon_energy_at_det)

    meta_data['q_tot'].append(summary['q_tot'])
    meta_data['n_channel'].append(summary['n_channel'])
    meta_data['n_channel_HLC'].append(summary['n_channel_HLC'])
    meta_data['muon_zenith'].append(most_energetic_track.dir.zenith)
    meta_data['muon_azimuth'].append(most_energetic_track.dir.azimuth)
    meta_data['muon_time'].append(most_energetic_track.time)
    meta_data['muon_pos_x'].append(most_energetic_track.pos.x)
    meta_data['muon_pos_y'].append(most_energetic_track.pos.y)
    meta_data['muon_pos_z'].append(most_energetic_track.pos.z)
    meta_data['spline_mpe_zenith'].append(spline_mpe.dir.zenith)
    meta_data['spline_mpe_azimuth'].append(spline_mpe.dir.azimuth)
    meta_data['spline_mpe_time'].append(spline_mpe.time)
    meta_data['spline_mpe_pos_x'].append(spline_mpe.pos.x)
    meta_data['spline_mpe_pos_y'].append(spline_mpe.pos.y)
    meta_data['spline_mpe_pos_z'].append(spline_mpe.pos.z)

def label_muons(file):
    i3file = file
    gcd = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz"
    tray = I3Tray()
    tray.Add("I3Reader", Filenamelist=[gcd, i3file])
    tray.Add(
    MCLabeler,
    event_properties_name=None,
    mctree_name='I3MCTree_preMuonProp',
    weight_dict_name='I3MCWeightDict',
    bg_mctree_name="I3MCTree_preMuonProp",
    )
    tray.Add(muonframe, Streams=[icetray.I3Frame.Physics])
    tray.Add(framestats, Streams=[icetray.I3Frame.Physics])
    tray.Execute()
    tray.PrintUsage()
    return meta_data, pulse_data, event_count

dataset_id = dataset_id
file_index_start = file_index_start
file_index_end = file_index_end
outdir = args.OUTDIR
os.makedirs(outdir, exist_ok=True)

directory = args.INDIR
pattern = args.INFILE_BASE
suffix = args.INFILE_SUFFIX
file_pattern = os.path.join(directory, pattern+'*')


i3files = sorted(glob.glob(file_pattern))
pulse_frames = []
meta_frames = []
total_event_count = 0
for i, i3file in enumerate(i3files):
    print(f"Processing file {i+1}/{len(i3files)}: {i3file}")
    meta_data, pulse_data, event_count = label_muons(i3file)
    df_pulses = pd.DataFrame.from_dict(pulse_data)
    df_meta = pd.DataFrame.from_dict(meta_data)

    pulse_frames.append(df_pulses)
    meta_frames.append(df_meta)
    print("Len of pframes", len(pulse_frames))
    print("Length of meta frames", len(meta_frames))
    total_event_count += event_count

df_pulses = pd.concat(pulse_frames).reset_index(drop=True)
df_meta = pd.concat(meta_frames).reset_index(drop=True)

ofile_pulses = os.path.join(outdir, f"pulses_ds_{dataset_id}_from_{file_index_start}_to_{file_index_end}_10_to_100TeV.ftr")
ofile_meta   = os.path.join(outdir, f"meta_ds_{dataset_id}_from_{file_index_start}_to_{file_index_end}_10_to_100TeV.ftr")

df_pulses.to_feather(ofile_pulses, compression='zstd')
df_meta.to_feather(ofile_meta, compression='zstd')
print(f"Stored {total_event_count} events in outfiles:\n  {ofile_pulses}\n  {ofile_meta}")