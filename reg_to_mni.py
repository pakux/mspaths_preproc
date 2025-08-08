#! env python3
# -*- coding: utf-8 -*-

__author__ = "Paul Kuntke"
__email__ = "paul.kuntke@uniklinikum-dresden.de"
__license__ = "BSD 3 Clause"

__derivative__ = "reg_to_mni"
__version__ = "0.1"
__longitudinal__ = False

from os.path import abspath
from nipype import IdentityInterface, MapNode, Node, Workflow
from nipype.interfaces.utility import Rename, Split, Merge

from nipype.interfaces.ants import RegistrationSynQuick, BrainExtraction
from nipype.interfaces.fsl.maths import ApplyMask

from wmi_nipype_workflows.register_to_mni import coregister_to_mni_wf

from wmi_nipype_workflows.reports import RegistrationRPT, SegmentationRPT
from wmi_nipype_workflows.wmi_workflow import WmiWorkflow


"""
This workflow registers Images into MNI-Space

"""

inputqueries = {
    "T1w": dict(
        suffix="T1w",
        space=None,
        extension="nii.gz",
        run=None,
        ceagent=None,
        acquisition=None,
        scope="raw",
    ),

    "FLAIR": dict(space=None, 
        suffix="FLAIR",
        extension="nii.gz",
        run=None,
        ceagent=None,
        acquisition=None,
        scope="raw",
    ),
}


@WmiWorkflow
def gen_wf(available_inputs=[], report_dir="/home/paulkuntke/mspaths/report"):
    """
    Create actual WmiWorkflow
    """

    outputfields = [
        "anat_@t1w",
        "xfm_@t1w2mni",
        "xfm_@affine2mni",
        "anat_@t1wmnispace",
        "anat_@imagemnispace",
        "anat_@t1wskullstripped",
        "anat_@brainmask"
    ]

    register =  MapNode(
        RegistrationSynQuick(transform_type="a", precision_type="float"),
        iterfield=['moving_image'],
        name="register",
    )

    t1w_split = Node(Split(splits=[1], squeeze=True), name="t1w_split")  # Squeeze T1w
    flair_split = Node(Split(splits=[1], squeeze=True), name="flair_split")  # Squeeze FLAIR

    brainextract = Node(
        BrainExtraction(
            brain_probability_mask=abspath("templates/OASIS/T_template0_BrainCerebellumProbabilityMask.nii.gz"),
            brain_template=abspath("templates/T_template0.nii.gz"),
            extraction_registration_mask=abspath("templates/OASIS/T_template0_BrainCerebellumRegistrationMask.nii.gz")
        ),
        name="brainextract",
    ) 
    brainmask_merger = Node(Merge(1), name="brainmask_merger")
    applymask_flair = MapNode(ApplyMask(), iterfield=["in_file"], name='applymask_flair')

    rename_image = MapNode(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_space-MNI152_FLAIR",
            keep_ext=True,
        ),
        iterfield=["in_file"],
        name="rename_image",
    )
    rename_t1w_warpfile = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_from-T1w_to-MNI152_desc-forward_xfm",
            keep_ext=True,
        ),
        name="rename_t1w_warpfile",
    )

    rename_t1w_affine = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_from-T1w_to-MNI152_desc-forward_xfm",
            keep_ext=True,
        ),
        name="rename_t1w_affine",
    )

    rename_t1w = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_space-MNI152_T1w",
            keep_ext=True,
        ),
        name="rename_t1w",
    )

    rename_skullstripped = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_desc-brain_T1w",
            keep_ext=True,
        ),
        name="rename_skullstripped"
    )
    rename_brainmask = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_desc-brain_mask",
            keep_ext=True,
        ),
        name="rename_brainmask"
    )
    outputnode = Node(IdentityInterface(fields=outputfields), name="outputnode")

    registration_wf = coregister_to_mni_wf(skullstripped=True, coreg_masks=True)
    
    gen_wf.wf = Workflow(name=__derivative__)
    gen_wf.wf.connect(
        [
            # Register FLAIR to T1W-Space
            (gen_wf.inputnode, t1w_split, [('T1w', 'inlist')]),
            (gen_wf.inputnode, register, [('FLAIR', 'moving_image')]),
            (t1w_split, register, [('out1', 'fixed_image')]),
            (t1w_split, brainextract, [("out1", "anatomical_image")]),
            (
                brainextract,
                registration_wf,
                [
                    ("BrainExtractionBrain", "inputnode.t1"),
                ],
            ),
            # T1w
            (
                gen_wf.inputnode,
                rename_t1w,
                [("subject", "subject"), ("session", "session")],
            ),
            (registration_wf, rename_t1w, [("outputnode.t1_registered", "in_file")]),
            (rename_t1w, outputnode, [("out_file", "anat_@t1w")]),
            # Coregistered files
            (brainextract, applymask_flair, [('BrainExtractionMask', 'mask_file')]),
            (register, applymask_flair, [("warped_image", "in_file")]),
            (applymask_flair, registration_wf, [('out_file', 'inputnode.coreg_files')]),
            (brainextract, brainmask_merger, [('BrainExtractionMask', 'in1')]),
            (brainmask_merger, registration_wf, [('out', 'inputnode.coreg_masks')]),
            (                gen_wf.inputnode, rename_image,
                [("subject", "subject"), ("session", "session")],
            ),
            (registration_wf, rename_image, [("outputnode.coregistered", "in_file")]),
            (rename_image, outputnode, [("out_file", "anat_@imagemnispace")]),
            (brainextract, rename_skullstripped, [("BrainExtractionBrain", "in_file")]),
            (
                gen_wf.inputnode,
                rename_skullstripped,
                [("subject", "subject"), ("session", "session")],
            ),
            (rename_skullstripped, outputnode, [("out_file", "anat_@t1wskullstripped")]),
            (brainextract, rename_brainmask, [("BrainExtractionMask", "in_file")]),
            (
                gen_wf.inputnode,
                rename_brainmask,
                [("subject", "subject"), ("session", "session")],
            ),
            (rename_brainmask, outputnode, [("out_file", "anat_@brainmask")])
        ]
    )


    # Now Connect the Warpfile
    gen_wf.wf.connect(
        [
            # Warpfile
            (
                gen_wf.inputnode,
                rename_t1w_warpfile,
                [("subject", "subject"), ("session", "session")],
            ),
            (
                registration_wf,
                rename_t1w_warpfile,
                [("outputnode.warp_field", "in_file")],
            ),
            (
                rename_t1w_warpfile,
                outputnode,
                [("out_file", "xfm_@t1w2mni")],
            ),
            # Affine Matrix
            (
                gen_wf.inputnode,
                rename_t1w_affine,
                [("subject", "subject"), ("session", "session")],
            ),
            (
                registration_wf,
                rename_t1w_affine,
                [("outputnode.affine_matrix", "in_file")],
            ),
            (
                rename_t1w_affine,
                outputnode,
                [("out_file", "xfm_@affine2mni")],
            ),
        ]
    )


    ### Additionally create Registration Reports
    rprtt1w = Node(
        RegistrationRPT(contrast="T1w", command="reg_to_mni", report_dir=report_dir),
        name="rprtt1w",
    )
    rprtbrainmask = Node(
        SegmentationRPT(contrast="T1w", command="antsBrainExtraction", report_dir=report_dir),
        name="rprtbrainmask"
    )
    gen_wf.wf.connect(
        [
            (
                gen_wf.inputnode,
                rprtt1w,
                [
                    ("subject", "subject_id"),
                    ("session", "session_id"),
                ],
            ),
            (registration_wf, rprtt1w, [("inputnode.template", "in_file")]),
            (registration_wf, rprtt1w, [("outputnode.t1_registered", "moving_img")]),
            (
                gen_wf.inputnode,
                rprtbrainmask,
                [
                    ("subject", "subject_id"),
                    ("session", "session_id"),
                ],
            ),
            (t1w_split, rprtbrainmask, [("out1", "in_file")]),
            (brainextract, rprtbrainmask, [("BrainExtractionMask", "mask")]),
        ]
    ) 