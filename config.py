import yaml


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load config.yaml
config = load_config()


ALIBABA_CLOUD_ACCESS_KEY_ID = config['credentials']['ALIBABA_CLOUD_ACCESS_KEY_ID']
ALIBABA_CLOUD_ACCESS_KEY_SECRET = config['credentials']['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
ENDPOINT = config['api']['ENDPOINT']
BUCKET_NAME = config['api']['BUCKET_NAME']
MODEL_PATH = config['model']['MODEL_PATH']
OUTPUT_VIDEO_PATH = config['video']['OUTPUT_VIDEO_PATH']
SIMILARITY_THRESHOLD = config['video']['SIMILARITY_THRESHOLD']
EMAIL_REG = config['email']['EMAIL_REG']
CODEC_FORMAT = config['video']['CODEC_FORMAT']