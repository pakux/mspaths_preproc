#! 




from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict



inputs:

    in_file 
    output
    disable_tta






run:

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
            )

