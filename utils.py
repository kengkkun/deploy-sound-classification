# Utils for preprocessing data etc
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import librosa
import numpy as np

input_length = 16000 * 5

n_mels = 320
batch_size = 32

base_classes = ['0_car-accident',
                '1_car-sudden-stop',
                '2_car-horns',
                '3_theft-alarm',
                '4_siren',
                '5_scream',
                '6_help',
                '7_fighting',
                '8_gun',
                '9_blast',
                '10_vehicle-engine',
                '11_train',
                '12_glass-broken',
                '13_residence-alarm',
                '14_fire-alarm',
                '15_dog-barkin',
                '16_gas-leaking',
                '17_crying-baby',
                '18_fire',
                '19_sound-at-night']

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        # "model_name": "efficientnet_model_1_10_classes" # change to be your model name
        "model_name": "sound_classification"  # change to be your model name
    },
    # "model_2": {
    #     "classes": sorted(base_classes + ["donut"]),
    #     "model_name": "efficientnet_model_2_11_classes"
    # },
    # "model_3": {
    #     "classes": sorted(base_classes + ["donut", "not_food"]),
    #     "model_name": "efficientnet_model_3_12_classes"
    # }
}


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """

    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    # turn input into list (ML Engine wants JSON)

    instances_list = instances.numpy().tolist()

    input_data_json = {
        "instances": instances_list}

    # print(model_path)
    # print(input_data_json)

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()

    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model


def preprocess_audio_mel_T(audio, sample_rate=16000, window_size=20,  # log_specgram
                           step_size=10, eps=1e-10):

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40

    return mel_db.T


def load_audio_file(file_path, input_length=input_length):

    # print(data)

    data, sr = tf.audio.decode_wav(
        file_path, desired_channels=1)

    
    print(data.shape)
    data = np.reshape(data, (data.shape[0],))

    # data = librosa.core.load(file_path, sr=16000, res_type='kaiser_fast', mono=True)[
    #     0]  # , sr=16000

   
    if len(data) > input_length:

        max_offset = len(data)-input_length

        offset = np.random.randint(max_offset)

        data = data[offset:(input_length+offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length -
                             len(data) - offset), "constant")

    data = preprocess_audio_mel_T(data)

    return data


# def load_and_prep_sound(filename, img_shape=224, rescale=False):
#     """
#     Reads in an image from filename, turns it into a tensor and reshapes into
#     (224, 224, 3).
#     """
#     # Decode it into a tensor
# #   img = tf.io.decode_image(filename) # no channels=3 means model will break for some PNG's (4 channels)
#     # make sure there's 3 colour channels (for PNG's)
#     img = tf.io.decode_image(filename, channels=3)
#     # Resize the image
#     img = tf.image.resize(img, [img_shape, img_shape])
#     # Rescale the image (get all values between 0 and 1)
#     if rescale:
#         return img/255.
#     else:
#         return img


def update_logger(sound, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns
    logger dictionary.
    """
    logger = {
        "sound": sound,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }
    return logger
