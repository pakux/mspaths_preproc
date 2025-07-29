#! env python3
# -*- coding: utf-8 -*-

__author__ = "Paul Kuntke"
__email__ = "paul.kuntke@uniklinikum-dresden.de"
__license__ = "BSD 3 Clause"

__derivative__ = "reg_to_mni"
__Version__ = "0.1"
__longitudinal__ = False


from nipype import IdentityInterface, MapNode, Node, Workflow
from nipype.interfaces.utility import Rename, Split

from nipype.interfaces.ants import RegistrationSynQuick

from wmi_nipype_workflows.register_to_mni import coregister_to_mni_wf
from wmi_nipype_workflows.reports import RegistrationRPT
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
def gen_wf(available_inputs=[], report_dir="tmp/report"):
    """
    Create actual WmiWorkflow
    """

    outputfields = [
        "anat_@t1w",
        "xfm_@t1w2mni",
        "xfm_@affine2mni",
        "anat_@t1wmnispace",
        "anat_@mrainmaskmnispace",
    ]

    register =  Node(
        RegistrationSynQuick(transform_type="a", precision_type="float"),
        name="register",
    )

    t1w_split = Node(Split(splits=[1], squeeze=True), name="t1w_split")  # Squeeze T1w
    flair_split = Node(Split(splits=[1], squeeze=True), name="flair_split")  # Squeeze FLAIR

    rename_image = MapNode(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_space-MNI152%(bidstags)s_%(suffix)s",
            keep_ext=True,
            parse_string=r"sub-.+_ses-.+_space-[\w|\d]+(?P<bidstags>(_\w+-[\w|\d]+)*)_(?P<suffix>\w+)_trans\.nii.*",
        ),
        iterfield=["in_file"],
        name="rename_image",
    )
    rename_t1w_warpfile = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_from-original_to-MNI152_desc-forward_xfm",
            keep_ext=True,
        ),
        name="rename_t1w_warpfile",
    )

    rename_t1w_affine = Node(
        Rename(
            format_string="sub-%(subject)s_ses-%(session)s_from-original_to-MNI152_desc-forward_xfm",
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

    outputnode = Node(IdentityInterface(fields=outputfields), name="outputnode")

    registration_wf = coregister_to_mni_wf(coreg_masks=True)
    gen_wf.wf = Workflow(name=__derivative__)
    gen_wf.wf.connect(
        [
            # Register FLAIR to T1W-Space
            (gen_wf.inputnode, t1w_split, [('T1w', 'inlist')]),
            (gen_wf.inputnode, flair_split, [('FLAIR', 'inlist')]),
            (t1w_split, register, [('out1', 'fixed_image')]),
            (flair_split, register, [('out1', 'moving_image')]),
            (
                gen_wf.inputnode,
                registration_wf,
                [
                    ("T1w", "inputnode.t1"),
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
            # Coregistered coreg_masks
            (
                gen_wf.inputnode,
                rename_mask,
                [("subject", "subject"), ("session", "session")],
            ),
            (
                registration_wf,
                rename_mask,
                [("outputnode.coregistered_masks", "in_file")],
            ),
            (rename_mask, outputnode, [("out_file", "anat_@maskmnispace")]),
            # Coregistered files
            (
                gen_wf.inputnode,
                rename_image,
                [("subject", "subject"), ("session", "session")],
            ),
            (registration_wf, rename_image, [("outputnode.coregistered", "in_file")]),
            (rename_image, outputnode, [("out_file", "anat_@imagemnispace")]),
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
                [("out_file", "xfm_@affinemdbrain2mni")],
            ),
        ]
    )

    ### Additionally create Registration Reports

    rprtt1w = Node(
        RegistrationRPT(contrast="T1w", command="reg_to_mni", report_dir=report_dir),
        name="rprtt1w",
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
        ]
    )
