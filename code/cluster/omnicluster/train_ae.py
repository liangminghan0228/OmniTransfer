import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import json
import configs
from configs import train_param_list, extra_padding
from utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dropout
import numpy as np

CONFIG_PATH = 'configs'

# @time_consuming("INIT AE MODEL")
def init_model(data_shape, layer_num, kernel_size_list, stride_list, output_padding_list, dropout_list, filters_list,
               activation, optimizer, loss, save_path):
    K.clear_session()
    model = tf.keras.Sequential()
    # try:
    #     model = tf.keras.models.load_model(filepath=save_path)
    # except Exception as e:
    #     print(e)
    model.add(Input(shape=data_shape))
    for i in range(layer_num):
        # model.add(layers.ZeroPadding2D(padding=(0, kernel_size_list[i]-1)))
        model.add(Conv_1D(filters=filters_list[i], kernel_size=kernel_size_list[i], stride=stride_list[i],
                            activation=activation, name=f"encode_{i}"))
        if i == layer_num - 1:
            model.add(Dropout(dropout_list[i], name="z"))
        else:
            model.add(Dropout(dropout_list[i]))
    for i in range(layer_num):
        if i == layer_num - 1:
            model.add(
                Conv_1D_Transpose(filters=data_shape[2],
                                    kernel_size=kernel_size_list[layer_num - 1 - i],
                                    stride=stride_list[layer_num - 1 -
                                                        i], output_padding=output_padding_list[i],
                                    activation=activation, name="recons_out"))
            c = 0
            for l in range(layer_num):
                c += stride_list[-(l+1)] ** l
            # model.add(layers.Cropping2D(cropping=(0, (kernel_size_list[i]-1)*c)))
        else:
            model.add(
                Conv_1D_Transpose(filters=filters_list[layer_num - 2 - i],
                                    kernel_size=kernel_size_list[layer_num - 1 - i],
                                    stride=stride_list[layer_num - 1 -
                                                        i], output_padding=output_padding_list[i],
                                    activation=activation, name=f"decode_{layer_num-1-i}"))

    model.compile(optimizer=optimizer, loss=loss)
    return model


# @time_consuming("TRAIN AE MODEL")
def train_ae_model(model, data, batch_size, epochs, validation_split, save_path):
    train_res = model.fit(data, data, batch_size=batch_size, epochs=epochs,
                          validation_split=validation_split, use_multiprocessing=False, workers=1, verbose=1)
    model.save(save_path)
    return train_res


def main(params):
    physical_devices = tf.config.list_physical_devices('GPU') 
    for dev in physical_devices: 
        tf.config.experimental.set_memory_growth(dev, True)
    train_ae_start_time = time()
    x_train_data = get_train_data(params["data_path"])

    kernel_size = params["kernel_size_list"][0]
    # x_train_data = np.pad(x_train_data, ((
    #     0, 0), (0, 0), (kernel_size-1+extra_padding, kernel_size-1+extra_padding)), 'reflect')
    x_train = np.expand_dims(x_train_data, 3)
    data_shape = x_train.shape[1:]
    
    model = init_model(data_shape, params["layers_num"], params["kernel_size_list"], params["stride_list"],
                       params["output_padding_list"], params["dropout_list"], params["filters_list"],
                       params["activation"],
                       params["optimizer"], params["loss"], params["init_ae_model_file"])
    model.summary()
    train_res = train_ae_model(model, x_train, params["batch_size"], params["epochs"], params["validation_split"],
                               params["init_ae_model_file"])

    get_save_z(
        x_train,
        params["init_ae_model_file"],
        "ae",
        params["init_ae_z_file"]
        # pathlib.Path(str(params["init_ae_z_file"]).replace('init_ae_z', f'init_ae_z_{i}_all'))
    )
    print(f'train_ae_time:{time() - train_ae_start_time}')
    x_train_recons = get_save_reconstruct(
        x_train,
        params["init_ae_model_file"],
        "ae",
        pathlib.Path(str(params["init_ae_recons_file"]))
        # pathlib.Path(str(params["init_ae_recons_file"]).replace('init_ae_reconstruct', f'init_ae_reconstruct_{i}_all'))
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = configs.config(CONFIG_PATH)
    paras = {"data_path": config.get('train_ae', 'data_path'),
             "record_path": pathlib.Path(config.get('train_ae', 'record_path')),
             "init_ae_model_file": pathlib.Path(config.get('train_ae', 'init_ae_model_file')),
             "init_ae_z_file": pathlib.Path(config.get('train_ae', 'init_ae_z_file')),
             "init_ae_recons_file": pathlib.Path(config.get('train_ae', 'init_ae_recons_file')),
             "layers_num": config.getint('train_ae', 'layers_num'),
             "kernel_size_list": json.loads(config.get('train_ae', 'kernel_size_list')),
             "stride_list": json.loads(config.get('train_ae', 'stride_list')),
             "output_padding_list": json.loads(config.get('train_ae', 'output_padding_list')),
             "filters_list": json.loads(config.get('train_ae', 'filters_list')),
             "dropout_list": json.loads(config.get('train_ae', 'dropout_list')),
             "activation": config.get('train_ae', 'activation'),
             "if_shuffle": config.getboolean('train_ae', 'if_shuffle'),
             "batch_size": config.getint('train_ae', 'batch_size'),
             "epochs": config.getint('train_ae', 'epochs'),
             "validation_split": config.getfloat('train_ae', 'validation_split'),
             "optimizer": config.get('train_ae', 'optimizer'),
             "loss": config.get('train_ae', 'loss'), }

    for train_param in train_param_list:
        params = copy.deepcopy(paras)
        params['layers_num'] = train_param['layers_num']
        params['kernel_size_list'] = train_param['kernel_size_list']
        params['stride_list'] = train_param['stride_list']
        params['output_padding_list'] = train_param['output_padding_list']
        params['filters_list'] = train_param['filters_list']
        params['dropout_list'] = train_param['dropout_list']
        name_key = f"layers_num_{params['layers_num']}_kernel_size_list_{list2str(params['kernel_size_list'])}_stride_list_{list2str(params['stride_list'])}_output_padding_list_{list2str(params['output_padding_list'])}_filters_list_{list2str(params['filters_list'])}_dropout_list_{list2str(params['dropout_list'])}_batch_size_{params['batch_size']}"
        print(f'name_key: {name_key}')
        params["record_path"] = paras["record_path"].parent / \
            name_key / paras["record_path"].name
        params["record_path"].parent.mkdir(parents=True, exist_ok=True)

        params["init_ae_model_file"] = paras["init_ae_model_file"] / name_key
        params["init_ae_model_file"].parent.mkdir(parents=True, exist_ok=True)

        params["init_ae_z_file"] = paras["init_ae_z_file"].parent / \
            name_key / paras["init_ae_z_file"].name
        params["init_ae_z_file"].parent.mkdir(parents=True, exist_ok=True)

        params["init_ae_recons_file"] = paras["init_ae_recons_file"].parent / \
            name_key / paras["init_ae_recons_file"].name
        params["init_ae_recons_file"].parent.mkdir(parents=True, exist_ok=True)
    # with open(paras["record_path"], "w") as f:
    #     f.writelines("------------------ start ------------------\n")
    #     for eachArg, value in paras.items():
    #         f.writelines(eachArg + " : " + str(value) + "\n")
    #     f.writelines("------------------- end -------------------\n")
        with open(params["record_path"], mode='w', encoding='utf-8') as record_file:
            for i in range(1, 2):
                params['epochs'] = 100
                main(params, i*params['epochs'])
