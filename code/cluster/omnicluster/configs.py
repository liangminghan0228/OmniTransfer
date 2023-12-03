import pathlib
import configparser
import yaml
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def config(path: str, default_config: str = 'default.conf') -> configparser.ConfigParser:
    config_parser = configparser.ConfigParser()
    config_path = pathlib.Path(path)
    default_config = config_path.joinpath(default_config)
    config_list = [default_config]
    for file in pathlib.Path(config_path).iterdir():
        if file.suffix == '.conf' and file.name != default_config.name:
            config_list.append(file)
    config_parser.read(config_list, encoding='utf-8')
    return config_parser

def load_train_param():
    train_param_path = os.path.join(project_path, 'configs', 'train_param.yaml')
    f = open(train_param_path, mode='r', encoding='utf-8')
    param_list = yaml.load(f.read(), Loader=yaml.FullLoader)
    return param_list
 
# if __name__ == '__main__':
#     load_train_param()
train_param_list = load_train_param()
extra_padding = 24