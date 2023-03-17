"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import os

# project imports
from SynthSeg.predict_denoiser import predict

# third-party imports
from ext.lab2im import utils


def validate_training(prediction_dir,
                      gt_dir,
                      models_dir,
                      validation_main_dir,
                      target_segmentation_labels,
                      input_segmentation_labels=None,
                      evaluation_labels=None,
                      step_eval=1,
                      min_pad=None,
                      cropping=None,
                      topology_classes=None,
                      sigma_smoothing=0,
                      keep_biggest_component=False,
                      n_levels=5,
                      nb_conv_per_level=2,
                      conv_size=3,
                      unet_feat_count=24,
                      feat_multiplier=2,
                      activation='elu',
                      skip_n_concatenations=0,
                      recompute=True):

    # create result folder
    utils.mkdir(validation_main_dir)

    # loop over models
    list_models = utils.list_files(models_dir, expr=['dice', '.h5'], cond_type='and')[::step_eval]
    # list_models = [p for p in list_models if int(os.path.basename(p)[-6:-3]) % 2 == 0]
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        # build names and create folders
        model_val_dir = os.path.join(validation_main_dir, os.path.basename(path_model).replace('.h5', ''))
        dice_path = os.path.join(model_val_dir, 'dice.npy')
        utils.mkdir(model_val_dir)

        if (not os.path.isfile(dice_path)) | recompute:
            loop_info.update(model_idx)
            predict(path_predictions=prediction_dir,
                    path_corrections=model_val_dir,
                    path_model=path_model,
                    target_segmentation_labels=target_segmentation_labels,
                    input_segmentation_labels=input_segmentation_labels,
                    min_pad=min_pad,
                    cropping=cropping,
                    topology_classes=topology_classes,
                    sigma_smoothing=sigma_smoothing,
                    keep_biggest_component=keep_biggest_component,
                    n_levels=n_levels,
                    nb_conv_per_level=nb_conv_per_level,
                    conv_size=conv_size,
                    unet_feat_count=unet_feat_count,
                    feat_multiplier=feat_multiplier,
                    activation=activation,
                    skip_n_concatenations=skip_n_concatenations,
                    gt_folder=gt_dir,
                    evaluation_labels=evaluation_labels,
                    recompute=recompute,
                    verbose=False)
