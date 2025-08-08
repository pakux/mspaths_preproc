#! 



from nipype.interfaces.base.core import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import traits, TraitedSpec
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
from os.path import basename
from nipype


inputs:

    in_file 
    output
    disable_tta






run:

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
            )

class HD_BET_BrainextractorInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True)

class HD_BET_BrainextractorOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, mandatory=True)



class HD_BET_Brainextractor(BaseInterface):
    input_spec = HD_BET_BrainextractorInputSpec
    output_spec = HD_BET_BrainextractorOutputSpec


    def _run_interface(self, runtime):

        maybe_download_parameters()
        predictor = get_hdbet_predictor(
            )


        output_filename = basename(self.inputs['in_file'] )

        hdbet_predict(self.inputs['in_file'], output_filename)


        self._results['out_file'] =

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._results['out_file']
        return outputs  

