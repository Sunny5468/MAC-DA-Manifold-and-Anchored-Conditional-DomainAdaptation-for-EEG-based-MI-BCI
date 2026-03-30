from models import ATCNet


model_dict = dict(
    ATCNet=ATCNet,
)


def get_model_cls(model_name):
    return model_dict[model_name]
