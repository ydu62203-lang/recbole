"""
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.quick_start import run_recbole
from recbole.quick_start import load_data_and_model
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


if __name__ == '__main__':

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='/root/RecBole-master/RecBole-master/saved/KGAT-Nov-30-2024_17-38-11.pth',
)

    # configurations initialization
    # config = Config(model='KGAT', dataset='ml-1m')


    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    # dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    # train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    #model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # resume from break point
    checkpoint_file = '/root/RecBole-master/RecBole-master/saved/KGAT-Nov-30-2024_17-38-11.pth'
    trainer.resume_checkpoint(checkpoint_file)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)
"""

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.quick_start import run_recbole
from recbole.quick_start import load_data_and_model
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

if __name__ == "__main__":

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="/root/RecBole-master/RecBole-master/saved/KGAT-Nov-30-2024_17-38-11.pth",
    )

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # When calculate ItemCoverage metrics, we need to run this code for set item_nums in eval_collector.
    trainer.eval_collector.data_collect(train_data)

    # model evaluation
    checkpoint_file = (
        "/root/RecBole-master/RecBole-master/saved/KGAT-Nov-30-2024_17-38-11.pth"
    )
    test_result = trainer.evaluate(test_data, model_file=checkpoint_file)
    print(test_result)
