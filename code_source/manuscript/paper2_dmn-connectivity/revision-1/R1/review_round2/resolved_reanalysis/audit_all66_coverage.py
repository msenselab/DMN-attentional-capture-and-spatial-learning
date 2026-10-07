#!/usr/bin/env python3
"""Geometry-only, full-cohort parcel coverage audit; never reads outcomes/BOLD values."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[6])));HERE=Path(__file__).resolve().parent
OUT=HERE/'coverage_qc';ATLAS=Path(__import__('os').environ.get('DMN_ATLAS_CACHE', str(__import__('pathlib').Path.home() / 'nilearn_data')) + '/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
LABELS=Path(__import__('os').environ.get('DMN_ATLAS_CACHE', str(__import__('pathlib').Path.home() / 'nilearn_data')) + '/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt')
SCORES=ROOT/'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/participant_scores_n33.csv'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    OUT.mkdir(exist_ok=True);s=pd.read_csv(SCORES,usecols=['subject','group']).sort_values('subject');assert len(s)==33
    labs=[x.split('\t')[1] for x in LABELS.read_text().splitlines() if x.strip()];atlas=nib.load(ATLAS);cache={};rows=[];fingerprints={str(ATLAS):sha(ATLAS),str(LABELS):sha(LABELS),str(SCORES):sha(SCORES)}
    for rec in s.itertuples():
        for run in [1,2]:
            mask=ROOT/f'derivatives/fmriprep/sub-{rec.subject}/func/sub-{rec.subject}_task-suppression_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
            if not mask.exists():mask=ROOT/f'derivatives/fmriprep/sub-{rec.subject}/func/sub-{rec.subject}_task-suppression_run-0{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
            im=nib.load(mask);key=(im.shape,tuple(im.affine.ravel()));fingerprints[str(mask)]=sha(mask)
            if key not in cache:
                a=image.resample_img(atlas,target_affine=im.affine,target_shape=im.shape,interpolation='nearest',force_resample=True,copy_header=False)
                cache[key]=np.rint(np.asanyarray(a.dataobj)).astype(np.int16)
            a=cache[key];m=np.asanyarray(im.dataobj)>0;c=np.bincount(np.where(m,a,0).ravel(),minlength=401)[1:];native=np.bincount(a.ravel(),minlength=401)[1:]
            for code,(count,total,label) in enumerate(zip(c,native,labs),1):rows.append({'subject':rec.subject,'group':rec.group,'run':run,'native_label':code,'physical_label':label,'network7':label.split('_')[2],'atlas_voxels_on_grid':int(total),'voxels_in_brain_mask':int(count),'mask_coverage_fraction':float(count/total) if total else None})
    d=pd.DataFrame(rows);assert len(d)==66*400 and d.subject.nunique()==33
    d.to_csv(OUT/'all66_parcel_coverage.csv',index=False);bad=d[d.voxels_in_brain_mask==0];bad.to_csv(OUT/'zero_coverage_parcel_runs.csv',index=False)
    omitted=sorted(bad.native_label.unique().astype(int).tolist());remaining=[x for x in range(1,401) if x not in omitted]
    default=[i for i,l in enumerate(labs,1) if l.split('_')[2]=='Default'];missing_default=sorted(set(default)&set(omitted))
    result={'status':'GEOMETRY_ONLY_COMPLETED_BEFORE_EMPIRICAL_INFERENCE','subjects':33,'runs':66,'parcels_nominal':400,'unique_functional_grids':len(cache),'zero_coverage_rows':len(bad),'zero_coverage_details':bad.to_dict(orient='records'),'zero_coverage_union_native_labels':omitted,'globally_complete_native_labels':remaining,'globally_complete_count':len(remaining),'Default91_missing_native_labels':missing_default,'Default91_preserved':not missing_default,'input_sha256':fingerprints,'audit_code_sha256':sha(__file__),'outcome_associations_inspected':False}
    (OUT/'coverage_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','globally_complete_native_labels']},indent=2))
if __name__=='__main__':main()
