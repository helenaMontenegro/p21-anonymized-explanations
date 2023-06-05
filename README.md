# Privacy-Preserving Generative Adversarial Network for Case-Based Explainability in Medical Image Analysis
Official repository of the paper [Privacy-Preserving Generative Adversarial Network for Case-Based Explainability in Medical Image Analysis](https://ieeexplore.ieee.org/document/9598877).

## Requirements
* tensorflow (version: 1.14.0 or above)
* scikit-learn (version: 1.2.1)
* h5py (version: 3.1.0)

The data is organized in a .hdf5 file with the following folders:
* **id** - identity annotations, where each image is assigned a value in [0, number of patients[, corresponding to the ID of the patient;
* **dis** - medical annontations, where each image is assigned a value in [0, number of different commorbities[;
* **set** - 0 if the image belongs to the training set, 1 for validation or and 2 for the test set
* **images** - array of images

## Classification Networks

To train the image anonymization methods, first, we need to train the identity and disease recognition models. The multiclass identity and disease recognition models can be obtained by running the script ``` python classification/multiclass_classification_task.py```, with the following parameters:

Name | Type | Required | Description
---- | ---- | -------- | -----------
--task | string | yes | Classification task to perform: "disease" or "identity" recognition
--data_file | string | yes | Name of the .hdf5 file with the data
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 300)
-- batch_size | int | no | Batch size during training (default: 64)

This script generates a model for identity/disease recognition, which is saved on the folder indicated in ```save_folder```.

The siamese identity recognition network, used in the Privacy-Preserving model with Siamese Identity Recognition (PP-SIR) can be obtained by running ``` python classification/siamese_identity_recognition.py``` with the following parameters:

Name | Type | Required | Description
---- | ---- | -------- | -----------
--data_file | string | yes | Name of the .hdf5 file with the data
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 300)
-- batch_size | int | no | Batch size during training (default: 64)

This script saves the weights of the identity recognition model, which is saved on the folder indicated in ```save_folder```.

## Privacy-Preserving Networks

The Privacy-Preserving model with Multiclass Identity Recognition (PP-MIR) can be obtained by running ```python anonymization/pp-mir.py``` 

Name | Type | Required | Description
---- | ---- | -------- | -----------
--infer | bool | no | Loads VAE and generates anonymized data if True
--data_file | string | yes | Name of the .hdf5 file with the data
--masks_file | string | yes | Name of the .hdf5 file with the segmentation masks
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 2000)
-- batch_size | int | no | Batch size during training (default: 64)
--dis_file | string | yes | Name of .h5 file with the disease model
--id_file | string | yes | Name of .h5 file with the multiclass identity model
--encoder_weights | string | infer=True | Name of file with the weights of the encoder
--decoder_weights | string | infer=True | Name of file with the weights of the decoder

The Privacy-Preserving model with Siamese Identity Recognition (PP-SIR) can be obtained by running ```python anonymization/pp-sir.py``` 

Name | Type | Required | Description
---- | ---- | -------- | -----------
--infer | bool | no | Loads VAE and generates anonymized data if True
--data_file | string | yes | Name of the .hdf5 file with the data
--masks_file | string | yes | Name of the .hdf5 file with the segmentation masks
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 4000)
-- batch_size | int | no | Batch size during training (default: 64)
--dis_file | string | yes | Name of .h5 file with the disease model
--id_weights | string | yes | Name of .h5 file with the weights of the siamese identity model
--encoder_weights | string | infer=True | Name of file with the weights of the encoder
--decoder_weights | string | infer=True | Name of file with the weights of the decoder

These scripts generate an .hdf5 file (with the same structure as the original data file) containing the anonymized images.

## Generation of Counterfactual Explanations

The model to generate counterfactual explanations using PP-MIR can be obtained by running ```python counterfactual_generation/multiclass_cf``` with the following parameters:

Name | Type | Required | Description
---- | ---- | -------- | -----------
--infer | bool | no | Loads VAE and generates anonymized data if True
--training_factuals | bool | no | Trains the PP-MIR model to anonymize images if True
--data_file | string | yes | Name of the .hdf5 file with the data
--masks_file | string | yes | Name of the .hdf5 file with the segmentation masks
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 1000)
-- batch_size | int | no | Batch size during training (default: 64)
--dis_file | string | yes | Name of .h5 file with the disease model
--id_file | string | yes | Name of .h5 file with the multiclass identity model
--encoder_weights | string | training_factuals=False | Name of file with the weights of the encoder
--decoder_weights | string | training_factuals=False | Name of file with the weights of the factual decoder
--cf_weights | string | infer=True | Name of file with the weights of the counterfactual decoder

The model to generate counterfactual explanations using PP-SIR can be obtained by running ```python counterfactual_generation/siamese_cf``` with the following parameters:

Name | Type | Required | Description
---- | ---- | -------- | -----------
--infer | bool | no | Loads VAE and generates anonymized data if True
--training_factuals | bool | no | Trains the PP-MIR model to anonymize images if True
--data_file | string | yes | Name of the .hdf5 file with the data
--masks_file | string | yes | Name of the .hdf5 file with the segmentation masks
--save_folder | string | yes | Folder where model will be saved
-- epochs | int | no | Number of epochs to train the network (default: 1000)
-- batch_size | int | no | Batch size during training (default: 64)
--dis_file | string | yes | Name of .h5 file with the disease model
--id_weights | string | yes | Name of .h5 file with the weights of the siamese identity model
--encoder_weights | string | training_factuals=False | Name of file with the weights of the encoder
--decoder_weights | string | training_factuals=False | Name of file with the weights of the factual decoder
--cf_weights | string | infer=True | Name of file with the weights of the counterfactual decoder


## Citation
```
@article{montenegro2021privacy,
  title={Privacy-preserving generative adversarial network for case-based explainability in medical image analysis},
  author={Montenegro, Helena and Silva, Wilson and Cardoso, Jaime S},
  journal={IEEE Access},
  volume={9},
  pages={148037--148047},
  year={2021},
  publisher={IEEE}
}
```

